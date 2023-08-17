# Anime Scene Dataset Preparation Guide

To get started, follow these steps:

- `frameList.json` contains the names of the movies along with their corresponding scene frame numbers to extract. It comprises a total of 1986 images from 9 Shinkai movies.

- Download the movies from the provided links and rename them as specified in the json file.

    | Name                                 | Size     | Duration | Frame        | Frames # | Link
    | :----------------------------------- | :------: | :------: | :----------: | :------: | :------------------------------------------:
    | The Place Promised in Our Early Days | 873 MB   | 01:30:20 | 1920 x 1080  | 270      | [Link](https://drive.google.com/drive/folders/1cpsQ9rsg9EkgIi683xPi33FDJ1Mecy8h)
    | 5 Centimeters Per Second            | 1.89 GB  | 01:02:45 | 1920 x 1080  | 246      | [Link](https://drive.google.com/drive/folders/1-VyZQVimmmH81BxD9pvdyI565jgebzWx)
    | A Gathering of Cats                 | 12.1 MB    | 00:01:02 | 1920 x 1080  | 5       | [Link](https://www.youtube.com/watch?v=qTkNzIiBXRs)
    | Children Who Chase Lost Voices       | 2.14 GB  | 01:56:07 | 1920 x 1072  | 525      | [Link](https://drive.google.com/drive/folders/1It254oVH5f0isp5NEskMqrm4eKn_llyt)
    | The Garden of Words                 | 1.08 GB  | 00:46:01 | 1920 x 1080  | 181      | [Link](https://drive.google.com/drive/folders/11Zmx3HhwqpWe_dwKtAaWd_Erbo2Uwg46)
    | Someone's Gaze                      | 61.0 MB  | 00:06:44 | 1930 x 1080  | 17       | [Link](https://ddl.animeout.com/public.php?url=nimbus.animeout.com/series/00RAPIDBOT/Dareka%20no%20Manazashi//[AnimeOut]%20Dareka%20no%20Manazashi%20[1080pp]%20[E9ED99BE][Commie][RapidBot].mkv)
    | Cross Road                          | 31.0 MB  | 00:01:58 | 1920 x 1080  | 22       | [Link](https://www.youtube.com/watch?v=AfbNS_GKhPw)
    | Your name                           | 1.63 GB  | 01:46:35 | 1920 x 1080  | 413      | [Link](https://drive.google.com/drive/folders/11ZVj2VZtpaFDULsxquB-Fxc9zEBYodXv)
    | Weathering with You                 | 2.67 GB  | 01:52:20 | 1920 x 1080  | 307      | [Link](https://drive.google.com/drive/u/0/folders/1b00Z4sXYbImU0MjKHk5j2VlJJXpcGSqd)

    For "A Gathering of Cats" and "Cross Road," both of which are sourced from YouTube, use the following commands to download them:
    ```bash
    pip install yt-dlp
    yt-dlp -f "bv*[ext=mp4]" https://www.youtube.com/watch?v=qTkNzIiBXRs -o "A Gathering of Cats.mp4"
    yt-dlp -f "bv*[ext=mp4]" https://www.youtube.com/watch?v=AfbNS_GKhPw -o "Cross Road.mp4"
    ```

- Extract frames from the downloaded movies use the following script:
    ```bash
    cd Anime_dataset

    python frame_extract.py frameList.json path/to/movie/folder path/to/save/extracted/data --crop
    ```
    - `--crop`: Optional. Use this flag to enable cropping and save frames (1080x1080) in the same folder. If not specific, original frames (1920x1080) will be saved in separate folders named after each movie.

- Perform any necessary post-processing, e.g., resizing or cropping, to tailor the dataset to your specific requirements.
