# https://stackoverflow.com/questions/8081676/how-to-download-a-youtube-video-using-the-youtubes-api

1. You need to install all the modules listed on top of the python file:
pip install json
pip install os
pip install requests
pip install yt_dlp
pip install ffmpeg-python
pip install opencv-python
pip install regex
pip uninstall moviepy


2. Please note that SABC has different channels for each language:
~ afrikaans = https://www.youtube.com/@SABCNuus
~ tshivenda = https://www.youtube.com/@SABCMafhungo
~ isiXhosa = https://www.youtube.com/@SABCIindaba
~ english = https://www.youtube.com/@sabcdigitalnews
~ isiZulu = https://www.youtube.com/@SABCIzindaba
~ tsonga = https://www.youtube.com/@SABCMahungu
~ seSotho || sepedi || setswana = https://www.youtube.com/@SABCDitaba/featured
~ Ditaba tsa Sesotho 
~ Ditaba tša Sepedi 
~ Dikgang tsa Setswana

3. Please make sure that publishedAfter date is less than publishedBefore
e.g publishedAfter = 2022-01-01T00:00:00Z
    publishedBefore = 2022-02-10T00:00:00Z

4. Removing prints on the code:
- Comment out line #72 - #75

5. Remove iframe
- Comment out #203 - #205