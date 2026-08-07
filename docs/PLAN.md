# Machine learning for Pakistani semi-classical music

**Principal Investigator**: [Faheem Sheikh](fahim.sheikh@gmail.com)


**Last Updated**: August 07, 2026

## Background

The Indian subcontinent has a centuries old tradition of classical music.
Informal and formal institutions, like the musical families (gharanas)[1] and art institutes, provide the means by which individuals can master this craft.
Muslim musicians have made numerous contributions to Indian (Hindustani) classical music dating back to the times of Amir Khusro[2]. 
Their influence helped in cultivating a rich, diverse and shared musical heritage throughout South Asia.
Post-partition, while classical music has maintained its place among a set of audiences in India, Pakistan's interest and contribution to this craft has been dwindling.
Primary reasons[3] for this decline are i) lack of patronage both on governmental and societal level, sometimes due to perceived bias that music is a non-islamic art form[4], ii) preference among classical musicians to remain in their shell, refusing to evolve their craft with time, and iii) indifference of academia and researchers towards music as a serious subject.
This last point is quite significant in the light of the observation that today classical music is increasingly being pursued by people having no prior musical background[5].
It includes professionals from South Asian diaspora in North America, Europe as well as western listeners lured by the feel of classical music.
This untraditional audience in their quest of classical music, demands more intricate, personalized information about the sounds they hear underscored by popularity of projects like Dream Journey[6].
Development of machine learning models benefitting emerging audiences to retrieve semantic information from classical music is a big motivation for this research proposal.

In the last few years, music information retrieval, most prominently under MIREX project[7] has been able to significantly benefit from advancement in machine learning and high performance computing.
Applied techniques from these subjects have given digital music applications like automatic recommendations, piracy control, genre (vocal) classification and instrument identification a whole new dimension.
Still the primary focus of automated music analysis and generation has been western music.
This is reflected in large musical datasets[8] available to researchers interested in machine understanding of music.
These datasets expose structural and perceptive properties of western music like pitch, harmony, chords and tempo.
While there are many similarities across western and non-western musical systems, crucially there are also significant differences e.g. Indian/Pakistani classical music has little or no harmonic component.
Ultimately, performance of a machine learning musical model has a direct correlation to the type and amount of musical patterns it has seen.
The aim of this project is therefore to tap into state-of-the-art music analysis techniques, use it to train or fine tune existing foundational music models with the aim to retrieve unique information from Pakistani Classical Music, and share it with listeners having little theoretical music knowledge in suitable digital format.

Attempts to make machines understand South Asian music have typically focused on learning the melodic content or raag as the first step[9].
The data set used in these studies is mostly personal collections of vocal artists or instrumentalists rendering various raags in traditional modes like khayal.
One example of such a dataset is the PIM-v1[10] which is used to train and explain how a neural network was able to learn raags in a similar way as human understanding.
However this dataset is almost exclusively based on vocal renditions captured from Indian public Radio archives [11].
A wider ranging data set covering both North and South Indian classical music is Dunya [12] covering many classical genres including instrumental music.
Still speaking with a geographical focus on Pakistan, one finds that South Asian classical music has evolved more in the direction of what can be termed as semi-classical music, which although rooted in raags needs more sophisticated analysis techniques because of the spiritual, regional and popular influences it contains[13].
The aforementioned datasets lack the representation of two primary Pakistani semi-classical genres including `Qawwali` and `Ghazal`.
The focus of this research proposal is on information retrieval from semi-classical music; nevertheless linking it with traditional classical music structures.
The route of understanding classical music by establishing similarity measures with semi-classical songs in sub-genres has been popularized by musicians[14] and archives[15] attempting to help people with little music theory background.
What remains to be seen is if this same route can also work for machine understanding of classical music.
A second contribution of this research proposal would be to find overlapping structures between Pakistani and western music.
To that end, this proposal would start from existing models trained exclusively on South Asian classical music to see how well they could learn Pakistani semi-classical music.
It will then take the observations to foundation models trained on western music with the goal to find and explain intricate similarities between western and South Asian music.

Overall, the project will attempt to evaluate the following hypotheses:
- Given excerpts from sub-genres of classical music popular in Pakistan, the existing AI models trained exclusively on Indian classical music are able to learn musical attributes like `raag`, `thaat` and `taal` with reasonable accuracy.
- The semantic information from western music foundation models and South Asian musical models could be fused to find similarities in music structures independent of their origin. These similarities are explainable by the resulting model in a comprehensible way for an ordinary listener.
- A unified machine learning model could be built to automatically tag Pakistani semi-classical music without any degradation in tagging accuracy for popular western music.

## Objectives

1. Construct a new dataset of Pakistani semi-classical music amenable for research dissemination under open source licensing. This data set will comprise short audio signals recorded from  scratch, converted from personal collections, or extracted from public music websites/portals after getting explicit license permissions.
2. Evaluate state of the art machine learning foundation models trained on the above data set. Publish the findings in leading Music information retrieval conferences and gather feedback on results.
3. Tune or extend with new layers the existing machine learning models to understand structural information of classical, semi-classical music with the aim of robust classification along raags, taals and sub-genre axes.
4. Develop an efficient AI pipeline which enables a user to dynamically classify a musical piece, helps it find other similar musical items for listening, and generate user friendly information explaining the musical piece under evaluation.
5. Optimize the resulting AI model for real-time inference on resource constrained computing devices.

## Methodology

South Asian classical music has had a large influence on movie soundtracks, semi-classical, folk and popular music throughout the Indian subcontinent.
Often untrained listeners are recommended to hear selected semi-classical pieces before they can appreciate the exclusively classic art form.
For instance if a movie song has been composed in a particular raag, after listening a human might be able relate it to a sitar only rendering of that raag.
Can an AI model learn classical music by mimicking this complex process in the human brain? Applied machine learning within music information retrieval domain can broadly be categorized under two approaches, the first relies on the extraction of spectral features like constant-Q transform or MEL Frequency Cepstral Coefficient (MFCC) and using them to train a multilayered convolution network[16].
The second approach, after the emergence of large-language models (LLM), is to train a foundation model on large samples of music followed by specialized re-training on a more limited and specific data set[17].

A common aspect in the exploration of either of these approaches is a dataset representing Pakistani semi-classical music.
Genres which need to be represented would be e.g. `Ghazals` favorite genre of expressing rich very popular Urdu poetry, movie songs from both Pakistani and Indian films of the 1950s/1960s (where compositions were still rooted into classical music), regional items like `Thumri`, `Tappa` etc. and spiritual music genres like `Qawwali` and `Kafi`.
In addition to representing the wide semi-classical spectrum, the audio samples should be amenable to experimentation, meaning these are of shorter duration, and there are no copy-right issues involved.
In line with established AI practices, the music samples will be divided into two parts, recordings where structural information of music is tagged by an expert musicologist serving as training data, while the rest of the samples will constitute a test data-set.
An example targeting only the Qawwali genre is the Qawwal-Rang dataset[18]. It would need to be significantly extended to cover other genres mentioned here in addition to labeling them with their melodic properties.

As a preparatory activity, the Principal Investigator developed a rules based classification method to recognize Qawwali genre[19].
Similarity metrics with audio samples in the popular western music dataset GTZAN were also published.
Afterwards a ResNet18 based convolutional neural network was successfully evaluated with the features from Qawwal-Rang dataset with over 90% accuracy[20].
These preliminary activities demonstrate the feasibility of the project as well as the potential of further work in this domain, including tasks such as raag/taal identification, explainability of music for untrained listeners and recommendation systems based on structural similarity.
To this end focus would be to use semantic information generated by a foundational music information model, and make it learn the note patterns and melodic structures specific to South Asian classical music by adding additional layers or attention heads.

Most interesting aspect of the project would be to compare the learning process of a successfully evaluated model with a trained classical musician.
This research proposal aims to collect ground truth not only containing labels/attributes but also free text rationalizing the musician's thought process.
This text would be used to train the reasoning part of a multimodal model
As a result the model will not only learn structural properties of the music samples but would also explain its decision in terms of the timestamps that contributed the most towards a particular tag.
This explainability part of the model would be most useful for untrained/casual listeners and the main reason for developing an audio-textual AI model.
It would be particular helpful in automatic tagging of Pakistani classical music where manual expert labeling would be hard to comeby with passage of time.

This research proposal also envisions development of an efficient inference engine capable of listening to live music or a recording and in real time producing decisions about genres, melodic and temporal structures in Pakistani semi-classical music.
For this part quantization, finetuning and optimization of the model would be required in order to deploy it on mobile phones or resource constrained devices.

## Expected Outcomes

This project will make theoretical as well as practical contributions to South Asian musicology.
Theory-wise, one of the aims is to find the extent to which an AI model can learn pure classical music based on information collected from semi-classical genres.
Second result will be an answer to how much of an overlap exists between computer based analysis of western and eastern music? Moreover, what enhancement in existing knowledge is required to develop robust and efficient algorithms for automatic identification of South Asian classical music sub-genres.
Lastly would be if an AI model is able to explain its decisions about musical structure with reasonable confidence matching the thought process of an expert musician?
On the practical front, by training an AI model to dynamically understand classical, semi-classical as well as folk music the information retrieval software will appeal to established players, music studios, academicians, budding musicians and students.
It can be a very useful experimentation tool for students and teachers of both humanities and computer engineering departments of universities interested in music from the Indian subcontinent.
Hopefully this software will act as a cross-pollination agent and new ideas will spring from these foundations.
But the most significant contribution of this project will be delivering a platform that incorporates a contemporary look on machine learning of South Asian classical music from the perspective of an untrained listener.
This will allow those researchers who do not understand music to start experimenting with the developed models using an AI first approach to learn more about music for which there is now a dearth of teachers.
This has the potential of exacerbating research efforts on classical music in India/Pakistan and has parallels in many other scientific fields.

## Evaluation and Dissemination
Data collected during the course of the project and software developed would be shared under an Open Source license on appropriate online platforms e.g. Hugging face and Github.
Here exact licensing model would be decided during due course.
A number of conference and journal publications in well established music information retrieval forums are expected as a result of this research proposal.

## Duration
The expected duration for data augmentation required to set up the research is expected to take between three to six months.
Evaluation of existing models, architecture adaptations for semi-classical genre detection is expected to last about six months.
Three more months are estimated for write up and submissions.
All estimates are based on availability of a full-time graduate student.
Once the genre detection project is successfully completed, some of the other tasks described in the research proposal are expected to follow a similar 12 months cycle
Detailed timeline breakdown could be provided upon request.

## Bibliography
[1] Kumar Prasad Mukherji, “The lost world of hindustani music”, Oxford University Press, Pakistan 2007.  
[2] Khan Muhammad Afzal, “Tafseel-e-Mausiqi (Explanation of Music)”, Anthology by Shahbaz Ali titled “Kya Soortain Hongi (Those were the Artists)” Sanjh Publications, 2012, Lahore, Pakistan.  
[3] Sadarang [Archives](http://www.sadarang.com/pakistani_music.htm)  
[4] Shaikh Aziz, “Musical heritage: The last rites?”, first appearance in DAWN newspaper, November 30,2003. Accessed from DAWN newspaper [archives](http://archives.dawn.com/weekly/images/archive/031130/images2.htm)  
[5] Jayashree Thatte Bhat, “Hindustani Vocal music: as seen outside India”, Abhinav publications, New Delhi, 2007  
[6] News [article](http://www.music-ir.org/mirex/wiki/MIREX_HOME) covering Dream Journey project  
[7] The music information retrieval exchange, [MIREX](http://www.music-ir.org/mirex/wiki/MIREX_HOME)    
[8] FMA: A Dataset for music [analysis](https://github.com/mdeff/fma)  
[9] P. M, K. T. Sreekumar, K. I. Ramachandran and C. S. Kumar, "Data Augmentation for Improving the Performance of Raga (Music Genre) Classification Systems," 2024 5th International Conference on Electronics and Sustainable Communication Systems (ICESC), Coimbatore, India, 2024, pp. 1407-1412, doi: 10.1109/ICESC60852.2024.10690091   
[10] P. Singh and V. Arora, "Explainable Deep Learning Analysis for Raga Identification in Indian Art Music," in IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 2302-2311, 2025, doi: 10.1109/TASLPRO.2025.3574839   
[11] PIM-v1 [dataset](https://github.com/ParampreetSingh97/PIM_v1_XAI/blob/main/Metadata.csv)    
[12] CompMusic Dunya [platform](https://github.com/MTG/dunya)    
[13] Yousuf Saeed, “Khayal Durpan: A documentary film” https://in.1947partitionarchive.org/node/884  
[14] Akhtar Ali Khan, Zakir Ali Khan,  “Norang-e-Mousiqi [Many Colors of Music]”, Urdu Science Board publication, Lahore, 2004.  
[15] [Sound of India](http://www.soundofindia.com)  
[16] S. Pasrija, S. Sahu and S. Meena, "Audio Based Music Genre Classification using Convolutional Neural Networks Sequential Model," 2023 IEEE 8th International Conference for Convergence in Technology (I2CT), Lonavla, India, 2023, pp. 1-5, doi: 10.1109/I2CT57861.2023.10126446.  
[17] M. Won, Y. -N. Hung and D. Le, "A Foundation Model for Music Informatics," ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea, Republic of, 2024, pp. 1226-1230, doi: 10.1109/ICASSP48485.2024.10448314.  
[18] Qawwal-Rang [DataSet](https://zenodo.org/records/6408796)     
[19] F. Sheikh, "[Qawwal Rang](https://medium.com/@fahim.sheikh/qawwalrang-a4fc0d2b2b59): An audio dataset for genre recognition of Qawwali"  
[20] Resnet18 based Qawwali [recognition](https://github.com/fsheikh/QawwalRang/pull/6)  

