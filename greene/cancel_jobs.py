import subprocess
job_id_end =  53427685
job_id_start = 53427625
for job_id in range(job_id_start,job_id_end):
    subprocess.run(["scancel",str(job_id)])

