FROM jferlez/fastbatllnn:deps
# switch to unpriviledged user, and configure remote access
USER ubuntu
WORKDIR /home/ubuntu
# Now copy over code
WORKDIR tools/FastBATLLNN
COPY --chown=ubuntu:ubuntu . .
RUN python3 posetFastCharm_numba.py
USER root
CMD /usr/local/bin/startup.sh