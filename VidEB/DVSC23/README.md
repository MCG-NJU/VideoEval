# DVSC23 evaluation

This is the codebase for the [2023 Video Similarity Challenge](https://sites.google.com/view/vcdw2023/video-similarity-challenge) and
the associated dataset.

The Video Similarity Challenge will be featured at the
[VCDW Workshop at CVPR 2023](https://sites.google.com/view/vcdw2023/video-similarity-challenge)!

The design and results of the challenge can be found in the
[paper](https://drive.google.com/file/d/1MujZDupmVVJC1h9GTU1LS4az-KoqbpZm/view).

## Running evaluations

### Descriptor eval

```
$ ./descriptor_eval.py --query_features vsc_eval_data/queries.npz --ref_features vsc_eval_data/refs.npz --ground_truth vsc_eval_data/gt.csv
Starting Descriptor level eval
...
2022-11-09 17:00:33 INFO     Descriptor track micro-AP (uAP): 0.4754
```


## License

The VSC codebase is released under the [MIT license](LICENSE).
