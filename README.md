# One Less Reason for Filter Pruning: Gaining Free Adversarial Robustness with Structured Grouped Kernel Pruning (SR-GKP)

> This is the official codebase for our NeurIPS 2023 paper ([OpenReview](https://openreview.net/forum?id=Pjky9XG8zP&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FConference%2FAuthors%23your-submissions))). Should you need to cite our paper, please use the following BibTeX:

```
@inproceedings{zhong2023adv_robust_gkp,
    title={One Less Reason for Filter Pruning: Gaining Free Adversarial Robustness with Structured Grouped Kernel Pruning},
    author={Shaochen Zhong and Zaichuan You and Jiamu Zhang and Sebastian Zhao and Zachary LeClaire and Zirui Liu and Daochen Zha and Vipin Chaudhary and Shuai Xu and Xia Hu},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

---

## Getting started with a runnable Colab Notebook

We provide a quick start [notebook](https://github.com/henryzhongsc/adv_robust_gkp/blob/main/tutorials/adv_robust_gkp_demo.ipynb) to demo how to prune a CIFAR-10-trained ResNet-20 according to SR-GKP specifications, then fine-tune the one-shot pruned model in a vanilla fashion.

---

## To-Do

- [x] Open source SR-GKP's implementation.
- [ ] Add paper highlight & results.
- [ ] Share reported model checkpoints (SR-GKP and others).
- [ ] Clean up and open source replication code for other iterative pruning methods evaluated in the paper.
- [ ] Migrate other one-shot pruning methods under our repo (following the same model definitions).