# ajout du service NGINX

Création du fichier nginx/nginx.conf
Création du fichier nginx/Dockerfile
Création du fichier nginx/README.md

Création du fichier /request.json

MAJ docker-compose :

  nginx:
    build:
      context: deployments/nginx/
      dockerfile: Dockerfile
    container_name: nginx_revproxy
    ports:
      - "8080:80"
      - "443:443"
    volumes:
      - ./deployments/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./deployments/nginx/certs:/etc/nginx/certs:ro
      - ./deployments/nginx/.htpasswd:/etc/nginx/.htpasswd:ro
    depends_on:
      - api

# Load Balancing
## Simuler la charge

Pour tester l'efficacité de notre configuration d'équilibrage de charge, nous devons simuler un grand nombre de requêtes.

### Outils de Test de Charge :

`ab` (ApacheBench) : Un outil simple en ligne de commande pour envoyer un grand nombre de requêtes HTTP. Utile pour des tests rapides.

### Monitoring : 

Pendant qu'on simule une charge, il est important de surveiller les logs de Nginx et de nos conteneurs API pour voir comment le trafic est distribué. 
Des outils de monitoring (comme Prometheus et Grafana, que nous verrons plus tard dans les modules) seraient idéaux pour visualiser la charge sur chaque instance.

Nous allons ici utiliser l'outil ab pour envoyer un grand nombre de requêtes et observer l'équilibrage de charge.

Commençons par installer la suite d'outils `apache2-utils` dont `ab` fait partie.

```
sudo apt-get install apache2-utils
```

Créons un fichier `request.json` à la racine de notre projet, contenant le contenu de nos requêtes :

```
{
    "petal_length": 6.5,
    "petal_width": 0.8
}
```

Puis exécutons le test suivant :

```
ab -n 1000 -c 100 -p request.json -T application/json http://localhost:8080/predict
```