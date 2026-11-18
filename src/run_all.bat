@echo off
rem Start ablation runs B-H in parallel, one minimized window per run.
rem Usage: run_all.bat [steps] [seed]
rem Baseline A (demo run, start it separately):
rem   ..\.venv\Scripts\python.exe main.py --total_steps 1000000 --seed 0

setlocal
cd /d "%~dp0"

set "STEPS=%~1"
if "%STEPS%"=="" set "STEPS=400000"
set "SEED=%~2"
if "%SEED%"=="" set "SEED=0"

set "PY=..\.venv\Scripts\python.exe"
set "COMMON=--total_steps %STEPS% --seed %SEED%"
set "LOGS=..\runs\console_logs"
if not exist "%LOGS%" mkdir "%LOGS%"

start "B single_q" /min cmd /c "%PY% main.py %COMMON% --single_q > %LOGS%\B_single_q.log 2>&1"
start "C no_smoothing" /min cmd /c "%PY% main.py %COMMON% --no_target_smoothing > %LOGS%\C_no_smoothing.log 2>&1"
start "D delay1" /min cmd /c "%PY% main.py %COMMON% --policy_delay 1 > %LOGS%\D_delay1.log 2>&1"
start "E delay4" /min cmd /c "%PY% main.py %COMMON% --policy_delay 4 > %LOGS%\E_delay4.log 2>&1"
start "F expl0.3" /min cmd /c "%PY% main.py %COMMON% --expl_noise 0.3 > %LOGS%\F_expl0.3.log 2>&1"
start "G expl0.03" /min cmd /c "%PY% main.py %COMMON% --expl_noise 0.03 > %LOGS%\G_expl0.03.log 2>&1"
start "H shaped" /min cmd /c "%PY% main.py %COMMON% --reward_shaping > %LOGS%\H_shaped.log 2>&1"

echo Started B-H: %STEPS% steps, seed %SEED%.
echo Logs: runs\console_logs   TensorBoard: tensorboard --logdir runs
endlocal
