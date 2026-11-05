Key links:
CS2243 Project repo
Libero repo
RD-VLA repo (use the same pretrained models as this paper)

Current project idea:
https://www.overleaf.com/project/6ab49acc512cae150ecd1ec1
TLDR: 
1. yea so a diffusion policy in general generates an action sequence/action chunk like A_t = (a_t, a_t+1,...,a_t+H)
 2. So in general we predict one of these chunks at a time. Tho we could stop executing mid chunk and generate a new chunk based off of new observations
 3. Each action is represented by an action token e.g. a_i
 4. During each denoising pass through the transformer (the transformer predicts how to denoise the current action chunk by e.g. estimating the noise. The proposed scheduler predicts which token group will benefit most from fresh FFN computation based on expected improvement in task outcome): all action tokens undergo attention which means token i can condition on other tokens
 5. Selected tokens get fresh FFN computations: the FFN processes the updated internal representations, which contributes to the transformer's denoising prediction.
 6. Other tokens reuse previously cached FFN outputs
 7. The question is how to choose which groups deserve fresh FFN computation based on the expected improvement in task outcome i.e. how to make this scheduler



Experiments to run next:
At what point of granularity will we run our variable noise schedule? Per-output dimension in action space (more MuLAN-esque) vs. per action (i.e. action-token a la Diffusion Forcing)
If MuLAN: how can we best incorporate the auxiliary latent with existing conditional vs. unconditional paradigms?
Can we purely posttrain vs. do we have to pretrain?
Simulated arm?

Foundational/prereq Knowledge

1 What is a diffusion policy? TODO
https://arxiv.org/abs/2303.04137
Transformer/U-net denoisers 



2) How to mix w/ transformers?
DiT: https://arxiv.org/abs/2212.09748

Lit Review


Ideas: what are pi_0 doing? 



Paper 1 (link): D3P: Dynamic Denoising Diffusion Policy via Reinforcement Learning
Executive summary of key ideas:


Tackles how iterative denoising is slow at inference time.


"Not every robot decision is equally important, so don't spend the same number of denoising steps on every decision."


D3P says robot behaviour has two kinds of moments: 1 Routine state (e.g. move through free space). 2 Critical state (e.g. grasp/precise manipulation)


So we first estimate how critical this state is, and then choose how many denoising steps to use.


D3P uses base policy \pi_{\theta} a lightweight state-aware adaptor K_{\omega} to choose the denoising budget, and trains the adaptor together with the diffusion policy using RL to trade off task performance against inference cost.
 2.2x speedup in simulation w/o comparable preformance success, and 1.9x acceleration on physical robot.


D3P might say: current situation is important, run 16 denoising iterations for this action chunk.


Our idea might ask a more granular thing: "within this chunk, a_4, a_5 correspond to grasp/contact and are more critical than the rest. Can I allocate diffusion differently across those trajectory positions?"


Use RL to jointly train the base diffusion policy and the adaptor (a base policy is fine-tuned with DPPO, and the lightweight adaptor is trained from scratch with PPO where the reward function incentivizes minimizing denoising steps without compromising performance).


The adapter predicts the noise-level strides: "large strides skip more noise levels to speed up routine actions, while small strides keep precise denoising for crucial actions."


They model the problem as a two-layer POMDP
 (outer layer = robot environment: state change is robot and object configuration - executes a completed action and changes the physical situation
 Inner layer = denoising process: the current noisy candidate action chunk - refines that candidate toward a usable action chunk)


They formulate denoising as a decision process so RL can assign credit to the computational decisions that produced an action, as well as to the final robot behaviour.


\bar{t}(t,i): t = decision number, i = current noise-level index.


They jointly train \pi_{\theta} and K_{\omega} on the same batch of rollout data.


K_{\omega} is a "Gaussian policy". Some networks can output a mean and standard deviation, and then you sample a stride.




Paper 2 (paper link, repo link): Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision–Language–Action Models via Latent Iterative Reasoning (Feb 2026)

Executive summary of key ideas:


It repeatedly applies the same transformer block to a latent representation inside the action head. The VLM backbone itself runs once per observation; its features condition the loop.
How it works:
Encode: A Qwen2.5-0.5B-based VLM processes images and instructions.
Prepare: A “Prelude” extracts context; a latent scratchpad starts from noise.
Loop: A weight-shared recurrent transformer updates the scratchpad, repeatedly attending to the context and robot state.
Decode: A “Coda” converts the refined representation into an action chunk.


I.e. this recurrent transformer is refining stuff, and then the refined representation is passed to Coda to generate actions. It stops refining (looping) when the representation stops changing much, using some delta threshold. 


A transformer can itself be the diffusion denoiser. The meaningful choice is therefore refine hidden features versus advance the diffusion process. (This paper does not do diffusion. However, in diffusion models, for every single noise level, you can do looping if you want!)


I.e. controller decides how many recurrent network passes to spend before advancing to the next noise level.




Paper 3 (link):  Looped Diffusion Transformer (Sep 2026)

Executive summary of key ideas:
Repeats shared transformer blocks within each denoising step for image generation; supervises intermediate loops. Studies spending compute on loop depth versus denoising steps.
Looped-DiT explicitly reports that extra loop depth can outperform additional denoising steps under a fixed inference budget. So “loop inside diffusion” alone is already established.
They fix the weak supervision of early loops by decoding every intermediate loop and supervise against the clean output, force intermediate states to be useful predictions
They also need to add inductive bias to help with training, “Exclusive Self Attention” mechanism to prevent attention from making the representations unstable
How does this apply to our project:
Direct precedent for combining latent refinement with diffusion and comparing the two compute axes.



^Proof that looped diffusion transformers work, now we apply to robotics^



Ideas:
- Via anchored idea: a noise model will denoise based on attention scores?






TODO for meeting:
Explain to a TF: What is our project?


“We learn where to loop”:
(Layer level): We learn a K_i where K_i is the amount of times you loop layer i
(Token level): We learn a token level mask/sigmoid layer that precedes layer K_i.
(Noise level) 



Test change of Layer level first
Test change of Token level
Test change of adding both



Plan: 1) Try with simple deterministic rules (e.g. K = 4)
2) Try to learn optimal K_i’s + preceding sigmoid layers



Namely:
Teacher forcing via Libero expert demonstration data
RL?


Questions:
The teacher forcing might require a loss based on the difference in outputted action probabilities vs action the expert took, but our RL rollout will only have the reward from success/failure of task. Is this justified? Is it kind of like a warm start before RL? 
Verify the above: will this work
What tasks do you recommend we do this on? A: anything w/ expert demos
Would hardware help? (beyond fun). Can we borrow Yilun lab hardware? Otherwise what hardware shld we ask for? What tasks: SO101 vs YAM arms?
How to scale to workshop/conference submission? A: more benchmarks
Need to ensure we fit in compute budget

Do we need to train the entire looped diffusion architecture ourselves? 


Will looping pick up on critical actions?\``	vb







Paper N (link): 

Executive summary of key ideas:

How does this apply to our project:




