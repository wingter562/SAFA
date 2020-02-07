ECHO OFF
ECHO ************************************************************************************************
ECHO Win batch script for running FL with varying lag_tolerance automatically
ECHO This script is supposed to run in the dir where FL.launcher.py locates
ECHO ************************************************************************************************

SET /P crash_prob=specify crash prob: 
ECHO preparing to test C = 0.1, 0.3, 0.5, 0.7, 1.0 with crash prob = %crash_prob%
:: ECHO preparing to test lag_t = 1 to 10 and C = 0.1 to 1.0 with crash prob = %crash_prob%

SET L=5
FOR %%C IN (0.3) DO (
  E:\programs\python\anaconda3\envs\federated_env\python.exe FL_launcher.py %crash_prob% %L% %%C
)  


:: finish
ECHO finished...
PAUSE