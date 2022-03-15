#### Attention-based Hybrid CNN-LSTM and Spectral Data Augmentation for COVID-19 Diagnosis from Cough Sound
This repository contains the used source code of the experiments which led to the results presented in the paper <b>Attention-based Hybrid CNN-LSTM and Spectral Data Augmentation for COVID-19 Diagnosis from Cough Sound</b>

We used <a href='https://www.nature.com/articles/s41597-021-00937-4'>COUGHVID</a> dataset. <a href='https://zenodo.org/record/4048312'>See in Zenodo</a>. <br> The augmented version will be available soon <a href="https://github.com/skanderhamdi/melspectrogram_spec_aug_30_percent_randomly_freq_time_masking">here</a>. <br>
The silence removed version of the dataset will be available soon <a href="https://github.com/skanderhamdi/coughvid">here</a>.

- <b>utils.py</b>: Provides som helpful functions (progressbar, plotROCCurve, plotCurves, plotConfusionMatrix,...)
- <b>pitch_shift.py</b>: Run this script to create the signal-augmented version of the dataset
- <b>spec_augment.py</b>: Run this script to create the spectral-augmented version of the dataset (apply SpecAugment technique) 
- <b>cnn_baseline.py</b>: Run this script to start CNN model training
- <b>lstm_baseline.py</b>: Run this script to start LSTM model training
- <b>cnn_lstm_baseline.py</b>: Run this script to start hybrid CNN-LSTM model training
- <b>attention_cnn_lstm.py</b>: Run this script to start Attention-based hybrid CNN-LSTM model training

<b>Note: </b> You should change the relative paths in the scripts

# Reference
Please cite our paper if you find this repository useful.

```
@article{hamdi2022,
  title = {Attention-based Hybrid CNN-LSTM and Spectral Data Augmentation for COVID-19 Diagnosis from Cough Sound},
  author = {Skander Hamdi, Mourad Oussalah, Abdelouahab Moussaoui and Mohamed Saidi},
  journal = {},
  year = {2022}
}
```

# Contact
Feel free to text me in <a href="skander.hamdi@univ-setif.dz">skander.hamdi@univ-setif.dz</a> for any questions or issues in the repository.
