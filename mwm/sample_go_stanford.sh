find ./data/go_stanford/ -maxdepth 1 -mindepth 1 -type d -print0 \
  | shuf -z -n 50 \
  | xargs -0 -n1 basename \
  > ./data_splits/go_stanford/test/traj_names.txt