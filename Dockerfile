#
# Python Dockerfile
#
# https://github.com/ginrou/handwritten_classifier
#

# Pull base image.
FROM micktwomey/ipython3.4

# install python libraries
RUN pip3.4 install Flask

# update source code
RUN rm -rf /var/www
RUN git clone https://github.com/ginrou/handwritten_classifier /var/www

# post code commands
WORKDIR /var/www

EXPOSE 5000
ENTRYPOINT ["/usr/bin/python3.4", "app.py"]