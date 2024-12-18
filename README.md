# UniDemoiré: Towards Universal Image Demoiréing with Data Generation and Synthesis

<center>Zemin Yang<sup>1</sup>, <a href="https://yujingsun.github.io/">Yujing Sun</a><sup>2</sup>, Xidong Peng<sup>1</sup>, <a href="https://www.cs.hku.hk/index.php/people/academic-staff/smyiu/">Siu Ming Yiu</a><sup>2</sup>, <a href="https://yuexinma.me/">Yuexin Ma</a><sup>1</sup></center>

### [Project Page](https://yizhifengyeyzm.github.io/UniDemoire-page/) | Dataset | Paper

***

The generalization ability of SOTA demoiréing models is greatly limited by the scarcity of data. Therefore, we mainly face two challenges to obtain a universal model with improved generalization capability: To obtain a vast amount of **1) diverse** and **2) realistic-looking moiré data**. Notice that traditional moiré image datasets contain real data, but continuously expanding their size to involve more diversity is extremely time-consuming and impractical. While current synthesized datasets/methods struggle to synthesize realistic-looking moiré images.

![Pipeline](./static/images/Pipeline.png)

Hence, to tackle these challenges, we introduce a universal solution, **UniDemoiré**. The data diversity challenge is solved by collecting a more diverse moiré pattern dataset and presenting a moiré pattern generator to increase further pattern variations. Meanwhile, the data realistic-looking challenge is undertaken by a moiré image synthesis module. Finally, our solution can produce realistic-looking moiré images of sufficient diversity, substantially enhancing the zero-shot and cross-domain performance of demoiréing models.

***

## :hourglass_flowing_sand: To Do

- [ ] Release training code
- [ ] Release testing code
- [ ] Release dataset
- [ ] Release pre-trained models

