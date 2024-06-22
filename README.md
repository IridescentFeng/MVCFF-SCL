# MVCFF-SCL
# Usage
## CIFAR100-LT
```
python main_cifar.py --data /home/kin/cifar100/ \
  --cifaropt 100 --epochs 300 \
  --featdim 100 \
  --cl_views sim-sim
```

## CIFAR10-LT
```
python main_cifar.py --data /home/kin/cifar10/ \
  --cifaropt 10 --epochs 300 \
  --featdim 10 \
  --cl_views sim-sim
```
