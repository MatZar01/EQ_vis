search_dir=cfgs/experiments_V
for entry in "$search_dir"/*
do
  python3 main.py "$entry"
done