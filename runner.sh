search_dir=cfgs/Res50_HS
for entry in "$search_dir"/*
do
  python3 main.py "$entry"
done