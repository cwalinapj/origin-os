FROM nginx:alpine

# Copy static files
COPY index.html /usr/share/nginx/html/index.html

# Environment variable substitution script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
