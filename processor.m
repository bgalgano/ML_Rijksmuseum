%% Image Processor to get into EfficientNet Form

% Load Folders
Main_folder = 'C:\Users\Myself\Documents\MATLAB\5660617';
Input_folder = 'C:\Users\Myself\Documents\MATLAB\5660617\jpg2';
Output_folder = 'C:\Users\Myself\Documents\MATLAB\5660617\out_img';

% Load List of Filenames
cd jpg2
file_Addr = dir('*.jpg');
cd C:\Users\Myself\Documents\MATLAB\5660617

% Shrink and Save Images
for i = [1:112039]
    flnm = join([file_Addr(i).folder,'\',file_Addr(i).name]);
    art_img = imread(flnm);
    small_img = imresize(art_img,[56,56]);
    out_flnm = join([Output_folder,'\',file_Addr(i).name]);

    imwrite(small_img,out_flnm);
end