# Notebooks

Collection of [jupyter notebooks](https://jupyter-notebook.readthedocs.io/en/stable/) useful to inspect data.


1. Run the notebook on the remote server
    - `jupyter notebook --no-browser --port=8888`
    - copy the token from the commandline
2. Access the notebook through ssh on client 
    - `ssh -N -f -L localhost:8889:localhost:8888 remote_user@remote_host`

--------------------------

## ChatGPT

To access a Jupyter notebook running on a remote server, you need to set up a few things. Here's a step-by-step guide:

1. Start Jupyter notebook on the remote server

SSH into your remote server and start your Jupyter notebook with the following command:

```bash
jupyter notebook --no-browser --port=8888
```
This starts Jupyter notebook on port 8888 and with the --no-browser option, it won't attempt to open a browser window on the server.

2. Set up SSH tunneling

In order to access your Jupyter notebook, you need to set up SSH tunneling from your local machine to the remote server. This allows you to access the port on which Jupyter is running (port 8888 in this case) securely. You can do this with the following command on your local machine:

```bash
ssh -N -f -L localhost:8889:localhost:8888 remote_user@remote_host
```

In this command:
-N tells SSH that no remote commands will be executed.
-f requests SSH to go to the background just before command execution.
-L specifies that the given port on the local (client) host is to be forwarded to the given host and port on the remote side (localhost:8889:localhost:8888 means forward traffic from port 8889 on the local machine to port 8888 on the remote machine).
remote_user@remote_host is your username and the IP address or hostname of the remote server.
3. Access Jupyter notebook in your local web browser

Now, you can open your web browser on your local machine and navigate to localhost:8889. You should be able to see your Jupyter notebook running.

Note: You will need the token or password from the Jupyter notebook server output (when you started the Jupyter notebook in step 1) to log in. If you see something like token=4d9fe48e16b8f5f3a63a05fe595f6f0cebaa14b3b6185978 in the terminal output, then 4d9fe48e16b8f5f3a63a05fe595f6f0cebaa14b3b6185978 is your token.

Keep in mind that this setup doesn't persist if the server or Jupyter restarts. For persistent access, you might want to look into using a reverse proxy with a web server like Nginx or Apache, or use a JupyterHub server.

Also remember to secure your setup properly when exposing a Jupyter notebook to the internet. This includes setting strong passwords and ideally also using HTTPS.


