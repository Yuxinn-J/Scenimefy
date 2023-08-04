# Prepare Anime Scene Dataset

To get started, follow these steps:

- `frameList.json` contains the names of the movies along with their corresponding scene frame numbers.

- Download the movies from the provided links and rename them as specified in the json file.
    - The Place Promised in Our Early Days：https://drive.google.com/drive/folders/1cpsQ9rsg9EkgIi683xPi33FDJ1Mecy8h 
    - 5 Centimeters Per Second: https://drive.google.com/drive/folders/1-VyZQVimmmH81BxD9pvdyI565jgebzWx 
    - A Gathering of Cats: https://www.youtube.com/watch?v=qTkNzIiBXRs
    - Children Who Chase Lost Voices: https://drive.google.com/drive/folders/1It254oVH5f0isp5NEskMqrm4eKn_llyt 
    - The Garden of Words: https://drive.google.com/drive/folders/11Zmx3HhwqpWe_dwKtAaWd_Erbo2Uwg46  
    - Someone's Gaze: ? https://www.youtube.com/watch?v=KSwbnUQJGnA&t=33s 
    - Cross Road: https://www.youtube.com/watch?v=AfbNS_GKhPw&t=34s
    - Your name: https://drive.google.com/drive/folders/11ZVj2VZtpaFDULsxquB-Fxc9zEBYodXv 
    - Weathering with You: https://drive.google.com/drive/u/0/folders/1b00Z4sXYbImU0MjKHk5j2VlJJXpcGSqd
    
- Extract frames from the downloaded movies through the following code:
    ```python
    cd anime_dataset

    python frame_extract.py frameList.json path/to/movie/folder path/to/save/extracted/data
    ```
    a total of xxx images from the 9 Shinkai movies should be obtained.
    
- perform any necessary post-processing, such as cropping, to tailor the dataset to your specific requirements.