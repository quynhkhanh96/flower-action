pip install av
# Download HMDB51 data and splits from serre lab website
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar

# Extract and organize video data..
mkdir -p video_data test_train_splits
unrar e test_train_splits.rar test_train_splits
rm test_train_splits.rar
unrar e hmdb51_org.rar 
rm hmdb51_org.rar
mv *.rar video_data

for filename in video_data/*.rar; do
    # foldername="${filename%.*}"
    mkdir -p video_data/${filename%.*}
    unrar e video_data/$filename video_data/${filename%.*}
done

rm video_data/*.rar