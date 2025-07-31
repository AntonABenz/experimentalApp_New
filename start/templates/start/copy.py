import shutil

names = ["Consent","Demographics","Instructions","Practice1","Practice2","Practice3","Practice4","Practice5","Practice6","Practice7","EndOfIntro",]  # replace this with your actual list of names

template_file = 'MyPage.html'  # replace this with your actual template file name

for name in names:
    new_file_name = f'{name}.html'
    shutil.copy(template_file, new_file_name)