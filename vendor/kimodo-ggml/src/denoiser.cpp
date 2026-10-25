#include "../include/kimodo/result.hpp"
#include "denoiser.hpp"
#include "ggml_weights.hpp"
#include "motion_rep.hpp"
#include "diffusion.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

namespace kimodo::detail {
namespace {
constexpr int width=1024, heads=8, head_width=128, text_tokens=50, prefix_tokens=52, layers=16, text_width=4096;
denoiser_profile profile_state;
int64_t clock_us() { return profile_state.clock_us ? profile_state.clock_us() : 0; }

// Every tensor one Transformer graph makes, counted the way the builders
// below make them, so its context is sized from the model's shape and not
// from a guess. transformer_for() checks the count it gets against this one.
//   linear: mul_mat, repeat, add                                       3
//   norm:   norm, repeat, mul, repeat, add                             5
//   embed:  4 linears, silu, linear, 3 concats, repeat, add
//   layer:  qkv linear; per [head, batch] 3 views, 3 conts, scores, scale,
//           soft_max, transpose, cont, value product, transpose (13); per
//           batch heads-1 concats and a reshape; batch-1 concats; out linear,
//           add, norm, linear, gelu, linear, add, norm
//   output: view, cont, linear
constexpr size_t linear_tensors=3, norm_tensors=5, branch_tensors=13;
size_t embed_tensors() { return 4*linear_tensors + 1 + linear_tensors + 3 + 2; }
size_t layer_tensors(size_t batch) {
    return linear_tensors + batch*heads*branch_tensors + batch*heads + (batch-1) +
           linear_tensors + 1 + norm_tensors + linear_tensors + 1 + linear_tensors + 1 + norm_tensors;
}
size_t output_tensors() { return 2 + linear_tensors; }
size_t transformer_tensors(size_t batch) { return embed_tensors() + layers*layer_tensors(batch) + output_tensors(); }

ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *bias) {
    auto *y=ggml_mul_mat(ctx,w,x);
    // F32 parity takes precedence over Tensor Core throughput.  In
    // particular, do not let a Vulkan backend lower the accumulation
    // precision for the reference model.
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return ggml_add(ctx,y,ggml_repeat(ctx,bias,y));
}
ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale, ggml_tensor *bias) {
    auto *n=ggml_norm(ctx,x,1.e-5f); return ggml_add(ctx,ggml_mul(ctx,n,ggml_repeat(ctx,scale,n)),ggml_repeat(ctx,bias,n));
}
// A graph's leaves are its inputs and the weights it names; `used` counts the
// weight lookups (at least the distinct weights) for the graph's size.
ggml_tensor *weight(const ggml_motion_weights&w,std::string_view n,size_t &used) { auto*t=w.tensor(n); if(!t) throw std::runtime_error("missing GGML tensor: "+std::string(n)); ++used; return t; }
ggml_tensor *layer(ggml_context *ctx,ggml_tensor*x,const ggml_motion_weights&w,std::string_view p,int seq,int batch,size_t &used) {
    const std::string s(p); auto*qkv=linear(ctx,x,weight(w,s+"self_attn.in_proj_weight",used),weight(w,s+"self_attn.in_proj_bias",used));
    // Use explicit [head, batch] branches for the F32 reference graph.  The
    // packed 4-D variant is faster, but differs slightly across Vulkan
    // backends; this layout exactly matches the PyTorch tensor boundaries.
    auto head = [&](int block, int h, int b) {
        return ggml_view_2d(ctx, qkv, head_width, seq, qkv->nb[1],
            static_cast<size_t>(block*width)*sizeof(float) +
            static_cast<size_t>(b)*qkv->nb[2] +
            static_cast<size_t>(h*head_width)*sizeof(float));
    };
    std::vector<ggml_tensor *> batches;
    batches.reserve(static_cast<size_t>(batch));
    for (int b=0; b<batch; ++b) {
        std::vector<ggml_tensor *> joined_heads;
        joined_heads.reserve(heads);
        for (int h=0; h<heads; ++h) {
            auto *q=ggml_cont(ctx,head(0,h,b)), *k=ggml_cont(ctx,head(1,h,b)), *v=ggml_cont(ctx,head(2,h,b));
            auto *scores=ggml_mul_mat(ctx,k,q);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            auto *prob=ggml_soft_max(ctx,ggml_scale(ctx,scores,1.f/std::sqrt(float(head_width))));
            auto *value_product=ggml_mul_mat(ctx,prob,ggml_cont(ctx,ggml_transpose(ctx,v)));
            ggml_mul_mat_set_prec(value_product, GGML_PREC_F32);
            joined_heads.push_back(ggml_transpose(ctx,value_product));
        }
        auto *joined=joined_heads.front();
        for (int h=1; h<heads; ++h) joined=ggml_concat(ctx,joined,joined_heads[static_cast<size_t>(h)],0);
        batches.push_back(ggml_reshape_3d(ctx,joined,width,seq,1));
    }
    auto *a=batches.front();
    for (int b=1; b<batch; ++b) a=ggml_concat(ctx,a,batches[static_cast<size_t>(b)],2);
    a=linear(ctx,a,weight(w,s+"self_attn.out_proj.weight",used),weight(w,s+"self_attn.out_proj.bias",used)); x=norm(ctx,ggml_add(ctx,x,a),weight(w,s+"norm1.weight",used),weight(w,s+"norm1.bias",used)); auto*ff=linear(ctx,x,weight(w,s+"linear1.weight",used),weight(w,s+"linear1.bias",used)); ff=ggml_gelu_erf(ctx,ff); ff=linear(ctx,ff,weight(w,s+"linear2.weight",used),weight(w,s+"linear2.bias",used)); return norm(ctx,ggml_add(ctx,x,ff),weight(w,s+"norm2.weight",used),weight(w,s+"norm2.bias",used));
}

// One Transformer (root_model or body_model) as ONE graph: the embedding, the
// 16 layers and the output linear, the same ops in the same order as the
// graph per stage it replaces, so `state` never leaves the GPU between
// layers. It is built and allocated once for its (prefix, batch, frames) and
// run again on every diffusion step: only its inputs change. The inputs live
// in a buffer of their own (the graph allocator never reuses their bytes),
// and an input whose values did not change since its last upload is not sent
// again (the text embedding and the heading are the same on every step of a
// job, the position encoding on every step of every job of that length).
struct transformer_graph {
    const ggml_motion_weights *weights=nullptr; std::string prefix; size_t batch=0, frames=0, motion_dim=0, outdim=0;
    ggml_context *in_ctx=nullptr, *ctx=nullptr, *graph_ctx=nullptr;
    ggml_backend_buffer_t in_buf=nullptr; ggml_gallocr_t alloc=nullptr; ggml_cgraph *graph=nullptr;
    ggml_tensor *motion=nullptr,*text=nullptr,*time=nullptr,*angle=nullptr,*position=nullptr,*out=nullptr;
    std::vector<float> sent_text, sent_time, sent_angle;
    // Where each layer's nodes end in `graph` (the embedding's are in the
    // first layer's unit, the output's in the last), and the submits of the
    // current cut: `per_submit` layers each, as node lists over the one
    // allocation. us_per_layer: the slowest layer of the last pass (0: none yet).
    std::vector<int> layer_end;
    size_t per_submit=0; int64_t us_per_layer=0;
    ggml_context *submit_ctx=nullptr; std::vector<ggml_cgraph *> submits;
    transformer_graph()=default;
    transformer_graph(const transformer_graph&)=delete;
    transformer_graph &operator=(const transformer_graph&)=delete;
    ~transformer_graph() { if(submit_ctx)ggml_free(submit_ctx); if(alloc)ggml_gallocr_free(alloc); if(in_buf)ggml_backend_buffer_free(in_buf); if(graph_ctx)ggml_free(graph_ctx); if(ctx)ggml_free(ctx); if(in_ctx)ggml_free(in_ctx); }
};
// One graph per Transformer: a job of another batch or length replaces it.
thread_local std::vector<std::unique_ptr<transformer_graph>> graphs;
// Kimodo's root statistics, read from the device once per weights.
struct root_stats { const ggml_motion_weights *weights=nullptr; std::vector<float> gm, gs, lm, ls; };
thread_local root_stats stats_cache;

void upload(ggml_tensor *t, std::span<const float> v) {
    const int64_t t0=clock_us(); ggml_backend_tensor_set(t,v.data(),0,v.size()*sizeof(float));
    profile_state.upload_us+=clock_us()-t0; profile_state.upload_bytes+=int64_t(v.size()*sizeof(float));
}
void upload_changed(ggml_tensor *t, const std::vector<float> &v, std::vector<float> &sent) {
    if(sent.size()==v.size() && std::memcmp(sent.data(),v.data(),v.size()*sizeof(float))==0) return;
    upload(t,v); sent=v;
}

denoiser_submit_policy submit_policy;

// The pass as submits of `per_submit` layers each, node lists over the one
// allocation (cut at the layer ends, so a submit's nodes read only bytes the
// earlier submits wrote).
bool cut(transformer_graph &g, size_t per_submit) {
    if(g.per_submit==per_submit) return true;
    if(g.submit_ctx) ggml_free(g.submit_ctx);
    g.submit_ctx=nullptr; g.submits.clear(); g.per_submit=0;
    std::vector<std::pair<int,int>> ranges;
    for(size_t i=0;i<g.layer_end.size();i+=per_submit) {
        const int lo=i==0?0:g.layer_end[i-1], hi=g.layer_end[std::min(i+per_submit,g.layer_end.size())-1];
        ranges.emplace_back(lo,hi);
    }
    size_t bytes=0; for(auto [lo,hi]:ranges) bytes+=ggml_graph_overhead_custom(size_t(hi-lo),false);
    g.submit_ctx=ggml_init({bytes,nullptr,true}); if(!g.submit_ctx) return false;
    for(auto [lo,hi]:ranges) {
        auto *sg=ggml_new_graph_custom(g.submit_ctx,size_t(hi-lo),false);
        for(int n=lo;n<hi;++n) ggml_graph_add_node(sg,ggml_graph_node(g.graph,n));
        g.submits.push_back(sg);
    }
    g.per_submit=per_submit; return true;
}

kimodo::expected<transformer_graph *,std::string> transformer_for(const ggml_motion_weights&w,std::string_view prefix,size_t motion_dim,size_t batch,size_t frames) {
    for(auto &g:graphs) if(g->prefix==prefix) {
        if(g->weights==&w && g->batch==batch && g->frames==frames && g->motion_dim==motion_dim) return g.get();
        g.reset(); // another shape: free it before allocating the next
    }
    std::erase_if(graphs,[](const auto &g){return !g;});
    const int64_t t0=clock_us();
    auto g=std::make_unique<transformer_graph>(); g->weights=&w; g->prefix=std::string(prefix); g->batch=batch; g->frames=frames; g->motion_dim=motion_dim;
    const int seq=prefix_tokens+static_cast<int>(frames), B=static_cast<int>(batch);
    const std::string p(prefix);
    constexpr size_t inputs=5;
    g->in_ctx=ggml_init({inputs*ggml_tensor_overhead(),nullptr,true});
    if(!g->in_ctx)return kimodo::unexpected("GGML context allocation failed");
    g->motion=ggml_new_tensor_3d(g->in_ctx,GGML_TYPE_F32,int(motion_dim),int(frames),B);
    g->text=ggml_new_tensor_3d(g->in_ctx,GGML_TYPE_F32,text_width,text_tokens,B);
    g->time=ggml_new_tensor_3d(g->in_ctx,GGML_TYPE_F32,width,1,B);
    g->angle=ggml_new_tensor_3d(g->in_ctx,GGML_TYPE_F32,2,1,B);
    g->position=ggml_new_tensor_3d(g->in_ctx,GGML_TYPE_F32,width,seq,1);
    for(auto *t:{g->motion,g->text,g->time,g->angle,g->position}) ggml_set_input(t);
    g->in_buf=ggml_backend_alloc_ctx_tensors(g->in_ctx,w.backend());
    if(!g->in_buf)return kimodo::unexpected("GGML input buffer allocation failed");
    const size_t tensors=transformer_tensors(batch);
    g->ctx=ggml_init({tensors*ggml_tensor_overhead(),nullptr,true});
    if(!g->ctx)return kimodo::unexpected("GGML context allocation failed");
    auto*ctx=g->ctx; size_t used=0;
    auto*m=linear(ctx,g->motion,weight(w,p+"input_linear.weight",used),weight(w,p+"input_linear.bias",used)); auto*te=linear(ctx,g->text,weight(w,p+"embed_text.weight",used),weight(w,p+"embed_text.bias",used)); auto*ti=linear(ctx,g->time,weight(w,p+"embed_timestep.time_embed.0.weight",used),weight(w,p+"embed_timestep.time_embed.0.bias",used));ti=linear(ctx,ggml_silu(ctx,ti),weight(w,p+"embed_timestep.time_embed.2.weight",used),weight(w,p+"embed_timestep.time_embed.2.bias",used));auto*he=linear(ctx,g->angle,weight(w,p+"linear_first_heading_angle.weight",used),weight(w,p+"linear_first_heading_angle.bias",used));auto*x=ggml_concat(ctx,ggml_concat(ctx,ggml_concat(ctx,te,ti,1),he,1),m,1);x=ggml_add(ctx,x,ggml_repeat(ctx,g->position,x));
    std::vector<ggml_tensor *> layer_out;
    for(int i=0;i<layers;++i) { x=layer(ctx,x,w,p+"seqTransEncoder.layers."+std::to_string(i)+".",seq,B,used); layer_out.push_back(x); }
    auto*part=ggml_view_3d(ctx,x,width,int64_t(frames),B,x->nb[1],x->nb[2],size_t(prefix_tokens)*width*sizeof(float));part=ggml_cont(ctx,part);
    g->out=linear(ctx,part,weight(w,p+"output_linear.weight",used),weight(w,p+"output_linear.bias",used)); ggml_set_output(g->out);
    g->outdim=size_t(weight(w,p+"output_linear.bias",used)->ne[0]);
    size_t made=0; for(auto*t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) ++made;
    if(made!=tensors) return kimodo::unexpected("the Transformer graph made "+std::to_string(made)+" tensors, transformer_tensors() counts "+std::to_string(tensors));
    const size_t graph_size=made+used+inputs;
    g->graph_ctx=ggml_init({ggml_graph_overhead_custom(graph_size,false),nullptr,true});
    if(!g->graph_ctx)return kimodo::unexpected("GGML graph context allocation failed");
    g->graph=ggml_new_graph_custom(g->graph_ctx,graph_size,false); ggml_build_forward_expand(g->graph,g->out);
    // Every node of layer i+1 reads layer i's output, so in graph order layer
    // i's nodes end at its output node: the cut points between submits.
    const int nodes=ggml_graph_n_nodes(g->graph);
    for(int n=0,i=0;n<nodes && i<layers;++n) if(ggml_graph_node(g->graph,n)==layer_out[size_t(i)]) { g->layer_end.push_back(n+1); ++i; }
    if(g->layer_end.size()!=size_t(layers)) return kimodo::unexpected("the Transformer graph's layer outputs are not in its node order");
    g->layer_end.back()=nodes;
    g->alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend()));
    if(!g->alloc || !ggml_gallocr_reserve(g->alloc,g->graph) || !ggml_gallocr_alloc_graph(g->alloc,g->graph)) return kimodo::unexpected("GGML graph allocation failed");
    std::vector<float> position(size_t(seq)*width);
    for(int s=0;s<seq;++s)for(int d=0;d<width;d+=2){float z=float(s)*std::pow(10000.f,-float(d)/width);position[size_t(s)*width+d]=std::sin(z);position[size_t(s)*width+d+1]=std::cos(z);}
    profile_state.build_us+=clock_us()-t0; ++profile_state.builds;
    upload(g->position,position);
    graphs.push_back(std::move(g)); return graphs.back().get();
}
}

denoiser_profile &denoiser_profile_state() { return profile_state; }
denoiser_submit_policy &denoiser_submit_policy_state() { return submit_policy; }

void release_denoiser_graphs() { graphs.clear(); stats_cache=root_stats{}; }

std::size_t denoiser_graph_bytes() {
    size_t r=0; for(auto &g:graphs) r+=ggml_backend_buffer_get_size(g->in_buf)+ggml_gallocr_get_buffer_size(g->alloc,0); return r;
}

kimodo::expected<std::vector<float>, std::string> run_motion_transformer(const ggml_motion_weights&w,std::string_view prefix,std::span<const float> motion,size_t motion_dim,std::span<const float> embedding,std::span<const float> timesteps,std::span<const float> headings,size_t batch,size_t frames) try {
    if(!batch||!frames||motion.size()!=batch*frames*motion_dim||embedding.size()!=batch*text_width||timesteps.size()!=batch||headings.size()!=batch) return kimodo::unexpected("invalid Transformer input dimensions");
    const int64_t t_entry=clock_us();
    std::vector<float> text(batch*text_tokens*text_width),time(batch*width),angle(batch*2);
    for(size_t b=0;b<batch;++b) { std::memcpy(text.data()+b*text_tokens*text_width,embedding.data()+b*text_width,text_width*sizeof(float)); for(int d=0;d<width;d+=2){float z=timesteps[b]*std::pow(10000.f,-float(d)/width);time[b*width+d]=std::sin(z);time[b*width+d+1]=std::cos(z);} angle[2*b]=std::cos(headings[b]);angle[2*b+1]=std::sin(headings[b]); }
    auto g=transformer_for(w,prefix,motion_dim,batch,frames); if(!g)return kimodo::unexpected(g.error());
    upload((*g)->motion,motion);
    upload_changed((*g)->text,text,(*g)->sent_text); upload_changed((*g)->time,time,(*g)->sent_time); upload_changed((*g)->angle,angle,(*g)->sent_angle);
    // One submit for the pass, unless the policy has a budget: then the
    // first pass submits one layer at a time, and each later pass as many
    // layers per submit as the last pass's slowest layer fits in the budget.
    auto &gr=**g; const auto &pol=submit_policy; size_t per_submit=size_t(layers);
    if(pol.clock_us && pol.budget_us>0) per_submit=gr.us_per_layer>0 ? std::clamp<size_t>(size_t(pol.budget_us/gr.us_per_layer),1,size_t(layers)) : 1;
    if(!cut(gr,per_submit)) return kimodo::unexpected("GGML submit graph allocation failed");
    int64_t t0=clock_us(), slowest=0;
    for(auto *sg:gr.submits) {
        const int64_t s0=pol.clock_us?pol.clock_us():0;
        if(ggml_backend_graph_compute(w.backend(),sg)!=GGML_STATUS_SUCCESS) return kimodo::unexpected("GGML graph execution failed");
        if(pol.clock_us) slowest=std::max<int64_t>(slowest,(pol.clock_us()-s0)/int64_t(per_submit));
        ++profile_state.submits;
    }
    if(pol.clock_us) gr.us_per_layer=slowest;
    profile_state.compute_us+=clock_us()-t0; ++profile_state.graphs;
    std::vector<float> r((*g)->outdim*frames*batch);
    t0=clock_us(); ggml_backend_tensor_get((*g)->out,r.data(),0,r.size()*sizeof(float));
    profile_state.download_us+=clock_us()-t0; profile_state.download_bytes+=int64_t(r.size()*sizeof(float));
    if(profile_state.after_graph) profile_state.after_graph(profile_state.user);
    profile_state.total_us+=clock_us()-t_entry;
    return r;
} catch(const std::exception&e){return kimodo::unexpected(e.what());}

kimodo::expected<std::vector<float>, std::string> run_two_stage_denoiser(
    const ggml_motion_weights &weights, std::span<const float> x,
    std::span<const float> embedding, std::span<const float> timesteps,
    std::span<const float> headings, std::span<const float> mask,
    std::size_t batch, std::size_t frames) {
    const size_t dim=weights.motion_dim(), root_input_dim=2*dim, body_input_dim=2*dim-1;
    if (!batch || !frames || !dim || x.size()!=batch*frames*root_input_dim || mask.size()!=batch*frames)
        return kimodo::unexpected("invalid two-stage denoiser input dimensions");
    auto root=run_motion_transformer(weights,"root_model.",x,root_input_dim,embedding,timesteps,headings,batch,frames);
    if(!root)return kimodo::unexpected(root.error());
    if(stats_cache.weights!=&weights) {
        auto gm=weights.f32_values("stats.global_root.mean"), gs=weights.f32_values("stats.global_root.std"), lm=weights.f32_values("stats.local_root.mean"), ls=weights.f32_values("stats.local_root.std");
        if(!gm)return kimodo::unexpected(gm.error());
        if(!gs)return kimodo::unexpected(gs.error());
        if(!lm)return kimodo::unexpected(lm.error());
        if(!ls)return kimodo::unexpected(ls.error());
        stats_cache=root_stats{&weights,std::move(*gm),std::move(*gs),std::move(*lm),std::move(*ls)};
    }
    const auto &st=stats_cache;
    auto local=global_root_to_local_root(*root,mask,batch,frames,st.gm,st.gs,st.lm,st.ls);
    if(!local)return kimodo::unexpected(local.error());
    std::vector<float> body_input(batch*frames*body_input_dim);
    for(std::size_t b=0;b<batch;++b) for(std::size_t t=0;t<frames;++t) {
        const auto src=(b*frames+t)*root_input_dim, dst=(b*frames+t)*body_input_dim;
        std::memcpy(body_input.data()+dst,local->data()+(b*frames+t)*4,4*sizeof(float));
        std::memcpy(body_input.data()+dst+4,x.data()+src+5,(root_input_dim-5)*sizeof(float));
    }
    auto body=run_motion_transformer(weights,"body_model.",body_input,body_input_dim,embedding,timesteps,headings,batch,frames);
    if(!body)return kimodo::unexpected(body.error());
    std::vector<float> output(batch*frames*dim);
    for(std::size_t b=0;b<batch;++b)for(std::size_t t=0;t<frames;++t){const auto r=(b*frames+t)*5, q=(b*frames+t)*(dim-5), o=(b*frames+t)*dim;std::memcpy(output.data()+o,root->data()+r,5*sizeof(float));std::memcpy(output.data()+o+5,body->data()+q,(dim-5)*sizeof(float));}
    return output;
}

kimodo::expected<std::vector<float>, std::string> run_separated_cfg_denoiser(
    const ggml_motion_weights &weights, std::span<const float> motion,
    std::span<const float> embedding, float timestep, float text_weight,
    float constraint_weight, std::size_t frames) {
    return run_separated_cfg_denoiser_batch(weights, 1, motion, embedding, std::span<const float>(&timestep, 1),
                                            text_weight, constraint_weight, frames);
}

kimodo::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_batch(
    const ggml_motion_weights &weights, std::size_t prompts, std::span<const float> motion,
    std::span<const float> embedding, std::span<const float> timesteps, float text_weight,
    float constraint_weight, std::size_t frames) {
    const std::vector<float> empty(prompts*frames*weights.motion_dim(), 0.f), headings(prompts, 0.f);
    return run_separated_cfg_denoiser_conditioned_batch(weights, prompts, motion, embedding, empty, empty,
                                                        timesteps, headings, text_weight, constraint_weight, frames);
}

kimodo::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_conditioned(
    const ggml_motion_weights &weights, std::span<const float> motion,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, float timestep, float heading,
    float text_weight, float constraint_weight, std::size_t frames) {
    return run_separated_cfg_denoiser_conditioned_batch(weights, 1, motion, embedding, observed, observed_mask,
                                                        std::span<const float>(&timestep, 1), std::span<const float>(&heading, 1),
                                                        text_weight, constraint_weight, frames);
}

kimodo::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_conditioned_batch(
    const ggml_motion_weights &weights, std::size_t prompts, std::span<const float> motion,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, std::span<const float> timesteps, std::span<const float> heading,
    float text_weight, float constraint_weight, std::size_t frames) {
    const size_t dim=weights.motion_dim(), clip=frames*dim;
    if (!dim || !prompts || motion.size()!=prompts*clip || embedding.size()!=prompts*text_width || timesteps.size()!=prompts || !std::isfinite(text_weight) || !std::isfinite(constraint_weight))
        return kimodo::unexpected("invalid separated CFG denoiser input");
    if (observed.size()!=prompts*clip || observed_mask.size()!=prompts*clip || heading.size()!=prompts)
        return kimodo::unexpected("invalid separated CFG condition dimensions");
    for (size_t p=0;p<prompts;++p) if (!std::isfinite(timesteps[p]) || !std::isfinite(heading[p]))
        return kimodo::unexpected("invalid separated CFG denoiser input");
    // Each prompt is its own [text, constraint, unconditional] triple of
    // batch entries, prompt after prompt. Every operation in the Transformer
    // works per batch entry (attention runs one [head, batch] branch at a
    // time), so no prompt sees another.
    constexpr size_t cfg_batch=3; const size_t batch=prompts*cfg_batch;
    std::vector<float> extended(batch*frames*2*dim), text(batch*text_width), times(batch), headings(batch), mask(batch*frames,1.f);
    for(size_t p=0;p<prompts;++p) {
        const float *m=motion.data()+p*clip, *obs=observed.data()+p*clip, *om=observed_mask.data()+p*clip;
        for(size_t c=0;c<cfg_batch;++c) { times[p*cfg_batch+c]=timesteps[p]; headings[p*cfg_batch+c]=heading[p]; }
        for(size_t c=0;c<cfg_batch;++c) for(size_t t=0;t<frames;++t) {
            auto *dst=extended.data()+((p*cfg_batch+c)*frames+t)*2*dim;
            std::memcpy(dst,m+t*dim,dim*sizeof(float));
            // Upstream separated CFG is [text, constraint, unconditional]. Only
            // the constraint branch receives observed motion and its feature mask.
            if (c==1) for (size_t d=0;d<dim;++d) dst[d]=m[t*dim+d]*(1.f-om[t*dim+d])+obs[t*dim+d]*om[t*dim+d];
            if (c==1) std::memcpy(dst+dim,om+t*dim,dim*sizeof(float));
        }
        // Only branch zero has text. Branch one is constraint-only; branch two is
        // unconditional. This is the upstream separated-CFG batch order.
        std::memcpy(text.data()+p*cfg_batch*text_width,embedding.data()+p*text_width,text_width*sizeof(float));
    }
    auto all=run_two_stage_denoiser(weights,extended,text,times,headings,mask,batch,frames);
    if(!all)return kimodo::unexpected(all.error());
    std::vector<float> result(prompts*clip);
    for(size_t p=0;p<prompts;++p) {
        const float *a=all->data()+p*cfg_batch*clip; float *r=result.data()+p*clip;
        for(size_t i=0;i<clip;++i) r[i]=a[2*clip+i]+text_weight*(a[i]-a[2*clip+i])+constraint_weight*(a[clip+i]-a[2*clip+i]);
    }
    return result;
}

kimodo::expected<std::vector<float>, std::string> sample_motion_from_noise(
    const ggml_motion_weights &weights, std::span<const float> initial,
    std::span<const float> embedding, std::size_t frames, unsigned steps,
    float text_weight, float constraint_weight) {
    if(initial.size()!=frames*weights.motion_dim()) return kimodo::unexpected("invalid initial motion noise dimensions");
    auto schedule=make_cosine_schedule(1000,steps); if(!schedule)return kimodo::unexpected(schedule.error());
    std::vector<float> state(initial.begin(),initial.end()), next(state.size());
    for(unsigned i=steps;i-->0;) {
        auto clean=run_separated_cfg_denoiser(weights,state,embedding,float(schedule->use_timesteps[i]),text_weight,constraint_weight,frames);
        if(!clean)return kimodo::unexpected(clean.error());
        auto stepped=ddim_step(*schedule,i,state.data(),clean->data(),next.data(),state.size());
        if(!stepped)return kimodo::unexpected(stepped.error());
        state.swap(next);
    }
    return state;
}

kimodo::expected<std::vector<float>, std::string> sample_motion_from_noise_conditioned(
    const ggml_motion_weights &weights, std::span<const float> initial,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, float heading, std::size_t frames,
    unsigned steps, float text_weight, float constraint_weight) {
    if(initial.size()!=frames*weights.motion_dim() || observed.size()!=initial.size() || observed_mask.size()!=initial.size())
        return kimodo::unexpected("invalid conditioned motion noise dimensions");
    auto schedule=make_cosine_schedule(1000,steps); if(!schedule)return kimodo::unexpected(schedule.error());
    std::vector<float> state(initial.begin(),initial.end()), next(state.size());
    for(unsigned i=steps;i-->0;) {
        auto clean=run_separated_cfg_denoiser_conditioned(weights,state,embedding,observed,observed_mask,
                                                           float(schedule->use_timesteps[i]),heading,text_weight,constraint_weight,frames);
        if(!clean)return kimodo::unexpected(clean.error());
        auto stepped=ddim_step(*schedule,i,state.data(),clean->data(),next.data(),state.size());
        if(!stepped)return kimodo::unexpected(stepped.error());
        state.swap(next);
    }
    return state;
}
}
