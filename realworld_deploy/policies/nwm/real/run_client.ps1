$GoalName = "pillar"  #若要采集目标图像，请设置要采集的目标名称

#---------------------------------------
$CondaEnv  = "mujoco"
$ServerHost = "127.0.0.1"
$ServerPort = 8000
$RobotIp = "192.168.11.200"
$CameraSn = "233522073186"
$CameraRes = "1280,720,30"
$GoalsDir = ".\goals"
$RunDir = ".\runs"
#-----------------------------------------

conda activate $CondaEnv

python .\nwm_real_infer_client_zh.py `
  --server-host $ServerHost `
  --server-port $ServerPort `
  --robot-ip $RobotIp `
  --camera-sn $CameraSn `
  --camera-resolution $CameraRes `
  --goals-dir $GoalsDir `
  --run-dir $RunDir `
  --collect-goal `
  --goal-name $GoalName `
  --frame-is-bgr `


# ============================================================
# 使用说明
#  先开启端口转发 ssh -p 16307 -L 8000:127.0.0.1:8000 root@connect.westd.seetacloud.com
#  然后在Windows Powershell中执行 powershell -ExecutionPolicy Bypass -File .\run_client.ps1
# ============================================================
