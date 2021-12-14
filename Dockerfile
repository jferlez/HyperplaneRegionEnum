FROM jferlez/fastbatllnn:deps
# switch to unpriviledged user, and configure remote access
USER ubuntu
WORKDIR /home/ubuntu/tools/FastBATLLNN
# Now copy over code
COPY --chown=ubuntu:root . .
# RUN python3 posetFastCharm_numba.py
USER root
CMD /usr/local/bin/startup.sh