# Anime Scene Dataset Guide

To get started, follow these steps:

- `frameList.json` contains the names of the movies along with their corresponding scene frame numbers to extract. It comprises a total of 2620 images (1720x1080) from 9 Shinkai movies.

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
    
- Extract frames from the downloaded movies use the following script:
    ```python
    cd anime_dataset

    python frame_extract.py frameList.json path/to/movie/folder path/to/save/extracted/data --crop
    ```
    - `--crop`: Optional. Use this flag to enable cropping and save frames the same folder. If not specific, original frames will be saved in separate folders named after each movie.

- Perform any necessary post-processing, e.g., resizing or cropping, to tailor the dataset to your specific requirements.

----
- All the things that need to be done in the future.

    - [ ] Someone's Gaze source
    - [ ] youtube-dl
    - [ ] sanity check