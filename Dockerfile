FROM jferlez/fastbatllnn:deps
# switch to unpriviledged user, and configure remote access
USER ubuntu
WORKDIR /home/ubuntu
RUN echo "export PATH=/usr/local/bin:$PATH\nexport LD_LIBRARY_PATH=/usr/local/lib\nexport COIN_INSTALL_DIR=/usr/local" >> /home/ubuntu/.bashrc
# Now copy over code
WORKDIR tools/FastBATLLNN
COPY . .