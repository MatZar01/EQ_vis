search_dir=cfgs/experiments
for entry in "$search_dir"/*
do
  python3 main.py "$entry"
done