export PYTHONPATH="/home/pcl/sqh/pysot/"
# export PYTHONPATH=/path/to/pysot:$PYTHONPATH

python tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth \
    --video demo/bag.avi

python -u ../../tools/test.py \
  --snapshot model.pth \
  --dataset VOT2018 \
  --config config.yaml

python -u tools/test.py \
  --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth \
  --dataset VOT2018 \
  --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml

python tools/eval.py      \
  --tracker_path ./results \
  --dataset VOT2018        \
  --num 10                  \
  --tracker_prefix 'model'