# Omni Model 入门札记

最近在研究 [SGLang Omni 的 RFC](https://github.com/sgl-project/sglang/issues/16546)，感谢桂神、帅还有超哥提出的整体架构。乘此良机，扩展下自身对 Omni 的认知。这篇文章一方面记录我理解的 Omni 架构，同时斗胆提出一个尚未完全验证的 Omni 音频输出的优化方案。

<div style="text-align: center;">
  <img src="./omni_architecture.png" alt="Omni Architecture" style="width:50%;">
</div>

## 架构概念

### codec token

Codec 是 Compressor（压缩器）和 Decompressor（解压缩器）的首字母组合而成的缩写，同时承担起了编码与解码的功能。在 encode 阶段，audio encoder 将原始的音频波形（模拟信号或高采样率的数字信号）转换成离散的 Tokens。而在 decode 阶段，Talker 会根据前序信息，生成 codec tokens，然后通过 audio decoder 将 codec tokens 还原回声音波形。通过神经网络的压缩，Codec 能在几乎不损失音质的前提下，把高频采样的音频信号压缩成极少量的 Token。

### 主流的音频模型架构

1. 音频编码（Audio Encoding）：将高频的原始音频波形输入到 Audio Encoder 得到 codec tokens。区别于传统的语音转文字方法（TTS/ASR），Audio Encoder 保留了远超出文字本身的信息，诸如说话人的音色、语速、情感甚至背景噪音。

2. 理解与推理（Thinker Decode）：codec tokens 会传递给 Thinker，一个 transformer 架构的组件，负责理解与推理。Thinker 对 codec tokens 进行 prefill，得到 last hidden states，而后进一步采样得到 text tokens。

3. 音频编码（Talker Decode）：Talker 接收来自 Tinker 的 text tokens 和 Hidden States，生成 codec tokens。

4. 音频解码（Audio Decoding）：将 codec tokens 输入到 Audio Decoder 得到音频波形（Waveform）。

### 为什么 Talker 需要接收 Thinker 的 Hidden States？

这是我理解了 Omni 模型后立刻产生的疑问，“为什么 Talker 需要接收 Thinker 的 Last Hidden States？”，换句话说，“既然都接受了 Last Hidden States，为什么还需要接受 text tokens？”。从逻辑上讲，Last Hidden States 应该已经包含了 text tokens 的信息，足够支撑 Talker 生成 codec tokens。抱着这个问题，我去查找了几种答案。其中一个我比较信服的观点是为了文字输出与音频输出之间的一致性。从产品的表现形态来看，我们希望用户在使用 Omni 模型听到声音的同时，也能够看到字幕。倘若我们让 Talker 不等待 Thinker 生成的 text tokens 就直接开始 decode codec tokens，想要让 Talker 采样出的 codec tokens 表现出的文字和 Thinker 采样到的 text tokens 一致，这是有一定技术难度的。而对用户而言，语音和文字输出略有一些区别尚可接收，类似于听到“我非常同意”，但是看到的文字输出是“我非常认同”，尚且无伤大雅。但是输出越来越长，这种不统一会对用户体验有不可忽略的影响。

但是，这也并不能说服我，为什么一定要让 Talker 等待 Thinker 生成 text tokens 之后再开始 decode codec tokens。一个并未经过验证的想法很快在我心中萌生，让 Talker 和 Thinker overlap decode。具体来说，现在的推理流程是：

voice encoder 生成 input codec tokens 后传递给 thinker，thinker 完成 prefill，得到 last hidden states，然后继续 decode，得到所有的 text tokens；接着把 last hidden states 和 text tokens 输入给 talker，talker decode 得到 output codec tokens，发送给 Codec Decoder，让 decoder 还原得到最终的音频。

而我设想的推理流程是：

voice encoder 生成 input codec tokens 后传递给 thinker，thinker 完成 prefill，得到 last hidden states。到这步，都和现在的主流设计一致。但是，我会将 last hidden states 发送给 talker 立刻开始 decode codec token，这样 thinker decode text token 的过程和 talker decode codec token 的过程会 overlap 起来。而且按照我的预期，talker decode codec token 会比 thinker decode text token 更快。这样可以经常拿着 talker 先产生的 codec token 去强行校验 thinker 产生的 text token。

这种设计，我想称其为跨模态的投机采样（Cross-modal Speculative Decoding），从优化 TTFT（或者说 time to first voice）的角度来感受，似乎更优。当然，我不确定是否有人这么尝试设计过模型并且进行过训练。我描述的模型架构设计可能已经被尝试过了，但是可能训不出来，所以没人这么做。不过，我真的在很认真思考怎么去优化这个系统的推理效率，即便训练上和工程实现上可能难度重重。

这些想法还完全没有经过 profile 来验证。有几个可以考虑的 drawback：

1. 难以训练：在训练阶段，如果允许 Talker 和 Thinker 独立从 Hidden States 中采样，Thinker 的 loss function 需要考虑到和 Talker 的 codec token 进行校验而回滚的惩罚。而 Talker 的 loss function，或者说训练目标会发生很大的变化。原本同时接收 hidden states 和 text tokens 的训练目标大概是“在给定词的情况下，还原出最自然的声学特征”，而只用 hidden states 的模型，会变成“将脑海中的想法还原为最自然的声学特征”。感受中训练目标更难，但是现在都追求 e2e 的训练，可能没我想的这么艰难。

2. 字幕回滚：如果 Talker 已经输出了相当长度的音频，而 Thinker 检测到了校验不一致，只能将 text 输出进行回滚，会影响一定的用户体验。当然，比起回滚音频还是强多了。

3. 工程实现：显然，考虑到投机采样本身的实现难度，为回滚字幕，其功能难度绝对不低。