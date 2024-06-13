# FIVR-5K
<img src="https://raw.githubusercontent.com/MKLab-ITI/FIVR-200K/master/banner.png" width="100%">


## Dataset format

* The video annotations are in file annotation.json that has the following format:
```bash
{
  "5MBA_7vDhII": {
    "ND": [
      "_0uCw0B2AgM",
      ...],
    "DS": [
      "hc0XIE1aY0U",
      ...],
    "CS": [
      "ydEqiuDiuyc",
      ...],    
    "IS": [
      "d_ZNjE7B4Wo",
      ...],
    "DA": [
      "rLvVYdtc73Q",
      ...],
  },
  ....
}
```


## Evaluation

1. Generation of the result file 
    * A file that contains a dictionary with keys the YT ids of the query videos and values another dictionary with keys the YT ids of the dataset videos and values their similarity to the query.

    * Results can be stored in a JSON file with the following format:
    ```bash
    {
      "wrC_Uqk3juY": {
        "KQh6RCW_nAo": 0.716,
        "0q82oQa3upE": 0.300,
          ...},
      "k_NT43aJ_Jw": {
        "-KuR8y1gjJQ": 1.0,
        "Xb19O5Iur44": 0.417,
          ...},
      ....
    }
    ```

   * An implementation for the generation of the JSON file can be found [here](https://github.com/MKLab-ITI/FIVR-200K/blob/1c0c093f29eea6c71f8f154408b2a371b74cb427/calculate_similarities.py#L79)

2. Evaluation of the results
    * Run the following command to run the evaluation:
    ```bash
    python evaluation.py --result_file RESULT_FILE --relevant_labels RELEVANT_LABELS
    ```

    * An example to run the evaluation script:
    ```bash
    python evaluation.py --result_file ./results/lbow_vgg.json --relevant_labels ND,DS
    ```
    
    * Add flag `--help` to display the detailed description for the arguments of the evaluation script
    
3. Evaluation on the three retrieval task
    * Provide different values to the `relevant_labels` argument to evaluate your results for the three visual-based retrieval task
    ```bash
    DSVR: ND,DS
    CSVR: ND,DS,CS
    ISVR: ND,DS,CS,IS
    ```
    * For the Duplicate Audio Video Retrieval (DAVR) task provide `DA` to the `relevant_labels` argument


## Citation
If you use FIVR-200K dataset for your research, please consider citing the papers:
```bibtex
@article{kordopatis2019fivr,
  title={{FIVR}: Fine-grained Incident Video Retrieval},
  author={Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon and Patras, Ioannis and Kompatsiaris, Ioannis},
  journal={IEEE Transactions on Multimedia},
  year={2019}
}
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details
