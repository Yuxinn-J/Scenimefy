# Anime Scene Dataset Preparation Guide

To get started, follow these steps:

- `frameList.json` contains the names of the movies along with their corresponding scene frame numbers to extract. It comprises a total of 2620 images from 9 Shinkai movies.

- Download the movies from the provided links and rename them as specified in the json file.

    | Name                                 | Size     | Length   | Frame        | Frames # | Link
    | :----------------------------------- | :------: | :------: | :----------: | :------: | :------------------------------------------:
    | The Place Promised in Our Early Days | 873 MB   | 01:30:20 | 1920 x 1080  | 435      | [Link](https://drive.google.com/drive/folders/1cpsQ9rsg9EkgIi683xPi33FDJ1Mecy8h)
    | 5 Centimeters Per Second            | 1.89 GB  | 01:02:45 | 1920 x 1080  | 325      | [Link](https://drive.google.com/drive/folders/1-VyZQVimmmH81BxD9pvdyI565jgebzWx)
    | A Gathering of Cats                 | 873 MB   | 01:30:20 | 1920 x 1080  | 14       | TODO
    | Children Who Chase Lost Voices       | 2.14 GB  | 01:56:07 | 1920 x 1072  | 616      | [Link](https://drive.google.com/drive/folders/1It254oVH5f0isp5NEskMqrm4eKn_llyt)
    | The Garden of Words                 | 1.08 GB  | 00:46:01 | 1920 x 1080  | 233      | [Link](https://drive.google.com/drive/folders/11Zmx3HhwqpWe_dwKtAaWd_Erbo2Uwg46)
    | Someone's Gaze                      | 61.0 MB  | 00:06:44 | 1930 x 1080  | 35       | [Link](https://www.animeout.xyz/dareka-no-manazashi/)
    | Cross Road                          | 21.7 MB  | 00:01:58 | 1920 x 1080  | 28       | [Link](https://www.youtube.com/watch?v=AfbNS_GKhPw)
    | Your name                           | 1.63 GB  | 01:46:35 | 1920 x 1080  | 506      | [Link](https://drive.google.com/drive/folders/11ZVj2VZtpaFDULsxquB-Fxc9zEBYodXv)
    | Weathering with You                 | 2.67 GB  | 01:52:20 | 1920 x 1080  | 428      | [Link](https://drive.google.com/drive/u/0/folders/1b00Z4sXYbImU0MjKHk5j2VlJJXpcGSqd)


- Extract frames from the downloaded movies use the following script:
    ```python
    cd anime_dataset

    python frame_extract.py frameList.json path/to/movie/folder path/to/save/extracted/data --crop
    ```
    - `--crop`: Optional. Use this flag to enable cropping and save frames (1080x1080) in the same folder. If not specific, original frames (1920x1080) will be saved in separate folders named after each movie.

- Perform any necessary post-processing, e.g., resizing or cropping, to tailor the dataset to your specific requirements.

- Contact

----
- All the things that need to be done in the future.

    - [ ] A Gathering of Cats source
    - [ ] sanity check
