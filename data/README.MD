# Downloader  

file 'downloader.sh' is a bash script to download files from the google storage. URLs follow the instructions. 

if other sources are being considerd, the URLs should be changed

to run the downloader use : 
    
    $bash downloader.sh
    
## direcotry structure  

after the script is done with the downloads the structure will contain the following folders : 

* train -> data : this will have the tairning samples. Ground truth for each sample will be in the directory and will be named `mask.png`
* test -> data : this folder contains test set for which there is no ground truth. 

