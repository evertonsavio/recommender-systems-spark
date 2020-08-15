# Spark_Recommender_Systems
Sistema de recomendação usando Apache Spark com MLlib. Linguagem de programação: Java.  
  
* Estrategia: Collaborative Filtering -> Matrix Factorization -> Alternating Least Squares (ALS)  
  
### Funcionamento do algoritmo para Big Data  
  
* Os dados usados como input contem 3 atributos {id do usuario, id do video assistido, e a proporcao do video assistido },
a proporcao do video assistido é o quanto o usuario assistiu daquele video e é usado com um rating.  
  
* Através da tecnica de ALS o modelo é criado a partir do histórico dos usuários e é feito um agrupamento não supervisionado.   
  
* O metodo do modelo entao é aplicado aos usuarios, o numero 5 e escolhido como numero de outros videos recomendados:  
```
Dataset<Row> recomendacoesParaUsuario = model.recommendForAllUsers(5);
```
    
* O metodo retorna um dataset onde possui o array de recomendacaoes com suas respectivas probabilidades para os proximos 5 videos para cada usuario.  
  


