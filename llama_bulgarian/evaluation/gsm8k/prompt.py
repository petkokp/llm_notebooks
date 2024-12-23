# The default gsm8k prompt from the CoT paper
# https://arxiv.org/pdf/2201.11903.pdf page 35.

# PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
# A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

# Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
# A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

# Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
# A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

# Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
# A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

# Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
# A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

# Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

# Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33."""

PROMPT = """Q: В горичката има 15 дървета. Работниците ще засадят дървета в горичката днес. След като са готови, ще има 21 дървета. Колко дървета засадиха работниците в горичката днес?
A: Започваме с 15 дървета. По-късно имаме 21 дървета. Разликата трябва да е броят на дърветата, които са засадили. Значи трябва да са засадили 21 - 15 = 6 дървета. Отговорът е 6.

Q: Ако има 3 коли на паркинга и пристигнат още 2 коли, колко коли има на паркинга?
A: Вече има 3 коли на паркинга. пристигат още 2. Сега има 3 + 2 = 5 коли. Отговорът е 5.

Q: Лия ​​имаше 32 шоколада, а сестра й имаше 42. Ако са изяли 35, колко парчета общо им остават?
A: Лия ​​имаше 32 шоколадови бонбона, а сестрата на Лия имаше 42. Това означава, че първоначално е имало 32 + 42 = 74 шоколадови бонбона. 35 са изядени. Така общо те все още имат 74 - 35 = 39 шоколада. Отговорът е 39.

Q: Джейсън имаше 20 близалки. Даде на Дени близалки. Сега Джейсън има 12 близалки. Колко близалки даде Джейсън на Дени?
A: Джейсън имаше 20 близалки. Тъй като сега има само 12, сигурно е дал останалите на Дени. Броят на близалките, които е дал на Дени, трябва да е бил 20 - 12 = 8 близалки. Отговорът е 8.

Q: Шон има пет играчки. За Коледа получи по две играчки от мама и татко. Колко играчки има сега?
A: Той има 5 играчки. Той получи 2 от мама, така че след това има 5 + 2 = 7 играчки. След това той получи още 2 от татко, така че общо има 7 + 2 = 9 играчки. Отговорът е 9.

Q: В стаята имаше девет компютъра. Всеки ден от понеделник до четвъртък бяха инсталирани още пет компютъра. Колко компютъра има сега в стаята?
A: Има 4 дни от понеделник до четвъртък. Всеки ден се добавят по 5 компютъра. Това означава, че са добавени общо 4 * 5 = 20 компютъра. В началото имаше 9 компютъра, така че сега има 9 + 20 = 29 компютъра. Отговорът е 29.

Q: Майкъл имаше 58 топки за голф. Във вторник той загуби 23 топки за голф. В сряда той загуби още 2. Колко топки за голф имаше в края на сряда?
A: Майкъл първоначално имаше 58 топки. Той загуби 23 във вторник, така че след това той има 58 - 23 = 35 топки. В сряда той загуби още 2, така че сега има 35 - 2 = 33 топки. Отговорът е 33."""
