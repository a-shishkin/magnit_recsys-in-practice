# magnit_recsys-in-practice

Собрать образ:<br>
docker build --pull --rm -f "Dockerfile" -t a-shishkin "."<br>

Запустить контейнер:<br>
docker run --rm -d -p 5000:5000 --name a-shishkin a-shishkin:latest<br>

После запуска контейнера можно загрузить файл по url http://localhost:5000/download_file?path=output_df.csv<br>
После загрузки начнется выполнение скрипта, результат сохраняется в файле `/magnit_recsys-in-practice/data/output_df.csv`<br>
Разделитель - ;<br>

Скопировать результат:<br>
docker cp a-shishkin:/magnit_recsys-in-practice/data/output_df.csv output_df.csv<br>

Файл с результатом имеет следующий вид:<br>
<picture><img src="images/scores.png"  width="70%" height="70%"></picture>

Показать содержимое файла:<br>
http://localhost:5000/show_data?path=output_df&type=csv<br>