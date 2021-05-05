# e4040-2021Spring-project

This is the repository of team AAAA for e4040-2021Spring-project. The team members are Weiminghui Ji, Shiyuan Zheng, Hongyi Zhou. 

# Descriptions of the project

Two problems are solved in the project. 

1. Different  GANs  on  MNIST  and  Fashion-MNIST: Construct GAN, CGAN, ACGAN and InfoGAN and train them on two datasets: MNIST and Fashion-MNIST. The goal is to make CGAN, ACGAN and InfoGAN learn to generate images conditioned on class labels, and make GAN learn generate images without label information. 
2. Reverse  Image  Caption: Construct CGAN and train it on dataset ``Oxford-102  flower  dataset". The goal is to make the CGAN learn to generate images based on texts. 

# How to run the file

./GANs_mnist: files for the problem 1. GAN.ipynb, CGAN.ipynb, ACGAN.ipynb and InfoGAN.ipynb contain the model structure, training process and evaluation for GAN, CGAN, ACGAN, InfoGAN respectively. If you run any of these files, three folders will be created: ./GANs_mnist/checkpoints (contains trained model), ./GANs_mnist/logs (contains training log), ./GANs_mnist/results (contains results during the training). Other files in this folder contains utility classes or functions.

./CGAN_flower: files for the problem 2. cgan_flower.ipynb contains the model structure, training process and evaluation for GAN for the Reverse Image Caption problem. final_generator_model_link.txt contains the trained model for this problem.

./evaluation_demo: Contains a demo of using FID to evaluate GANs.

./figures: Screenshots of assignment being run in the cloud.



# How to obtain the dataset

1. For the first problem, our code automatically fetches the dataset, so you do not need to prepare them manually.
2. For the second problem, please download the dataset using the link written in: ./CGAN_flower/dataset_link.txt. In the link, please download 3 folders of files: flowers, dataset, dictionary. Please put the 3 folders in the ./CGAN_flower.

# Organization of this project
The directory tree of the directory is shown below

```
|   E4040.2021Fall.AAAA.report.wj2309.sz2936.hz2700.pdf
|   README.md
|   
+---CGAN_flower
|   |   cgan_flower.ipynb
|   |   dataset_link.txt
|   |   final_generator_model_link.txt
|   |   
|   \---images
|       +---model structure
|       |       flower_discriminator.png
|       |       flower_generator.png
|       |       flower_text_encoder.png
|       |       
|       \---running time
|               flower-test1.jpg
|               flower-test2.jpg
|               loss-acc.png
|               run-time-image-200-epoch.png
|               run-time-image-79-epoch.png
|               
+---evaluation_demo
|       CGAN_Evaluation_FID_Demo.ipynb
|       evaluation.py
|       
+---figures
|       gcp_work_example_screenshot_1.png
|       gcp_work_example_screenshot_2.png
|       gcp_work_example_screenshot_3.png
|       
\---GANs_mnist
        ACGAN.ipynb
        CGAN.ipynb
        evaluation.py
        GAN.ipynb
        InfoGAN.ipynb
        ops.py
        utils.py
        

```