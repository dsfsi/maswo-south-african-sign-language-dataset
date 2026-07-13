for language in setswana sesotho sepedi isizulu isixhosa isiswati isindebele tshivenda xitsonga afrikaans english 
#Key   Una - AIzaSyBFK_8MQOKhaJ7rmBxMaNSAnDjEPFpImhY  date 2000-12-29T23:59:59Z thapelo - AIzaSyBvFyEdPQ4-nlOIZcCVArYUnA1sZ4VSN9I 
do
       export OUTPUT_ROOT_DIR=../../../../M2M-MT-files/Youtube_Downloads_2/
       export PUBLISH_B=""
       export PUBLISH_A="2000-12-29T23:59:59Z"
       export LANGUAGE=${language}
       export MAX_OUTPUTS=100000000
       export API_KEY=AIzaSyBFK_8MQOKhaJ7rmBxMaNSAnDjEPFpImhY
       export RERUN=0
       export RUN_META_OUTPUT_DIR=../../../../M2M-MT-files/Youtube-Meta_Files_2/
       export COOKIES=./cookies.txt
       python sabc_crawl_track_cookie_file.py --language ${LANGUAGE} --api_key ${API_KEY} --maxResults ${MAX_OUTPUTS} --download_dir ${OUTPUT_ROOT_DIR} --rerun ${RERUN} --meta_data_dir ${RUN_META_OUTPUT_DIR} --cookies ${COOKIES} --publishedAfter ${PUBLISH_A}
    #    python Youtube_Crawler_DSFSI.py --language ${language} --maxResults ${MAX_OUTPUTS} --download_dir ${OUTPUT_ROOT_DIR} --publishedBefore ${PUBLISH_B} --publishedAfter ${PUBLISH_A} 
done       
