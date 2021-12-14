FROM jferlez/fastbatllnn:deps
# switch to unpriviledged user, and configure remote access
USER ubuntu
WORKDIR /home/ubuntu
RUN echo "export PATH=/usr/local/bin:$PATH\nexport LD_LIBRARY_PATH=/usr/local/lib\nexport COIN_INSTALL_DIR=/usr/local" >> /home/ubuntu/.bashrc
# Now copy over code
WORKDIR tools/FastBATLLNN
COPY . .
USER root
RUN python3 posetFastCharm_numba.py
RUN chown -R ubuntu:ubuntu /home/ubuntu/tools
CMD /usr/local/bin/startup.sh