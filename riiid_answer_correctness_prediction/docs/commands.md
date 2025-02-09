# Commands

## File Copy from local to remove machine
```
# File copy.
$ scp local_file_path host_name:remote_file_path

# Directory copy
$ scp -r local_file_path host_name:remote_file_path
```

## File Copy from GCS to remove machine
```
$ gsutil -m cp -r gs://{my-bucket} ./data/
```
