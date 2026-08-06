from langchain_text_splitters import CharacterTextSplitter

text= """
Marcela Quintanilla-Dieck is a natural storyteller with a passion for health-related communications. Believing we are all connected through stories and narratives, Marcela aims to use her communications skills to promote positive health outcomes.

As a Masters of Science in Public Relations student at the Boston University School of Communications, Marcela embraced opportunities to collaborate with peers at the School of Public Health. She did this first as a student in “SB 733, Mass Communication and Public Health,” taught by COVID Corps mentor, Lynsie Ranker, and then as a COVID Corps member.

Marcelas first COVID Corps project was an ideal blend of health and communications. She was part of a team (along with Lynsie and fellow COVID Corps member, Sharon Casey) that created a fact sheet for the Boston Medical Centers COVID-19 Biorepository. The Biorepository (or Biobank) allows researchers to study COVID-19 using blood and tissue from patients with COVID-19 who agree to participate.

The teams task was to design a factsheet for potential participants that could explain what a biorepository is, how this particular initiative aimed to advance knowledge of COVID-19, and how participants samples and information would be protected. Marcela chose styles, colors, icons, and wording for the factsheet. Marcela understood that the words “biorepository” and “biobank” might be intimidating to some people, but that presenting the information with pleasant colors and simple words could make people realize that they could not only understand the information, but also had an opportunity to help advance the science surrounding COVID-19.

Multiple studies are now underway using data from the Biorepository, and the team at BMC acknowledges the role of this project: “These materials were a critical component to our patient recruitment strategy, and we were so thankful for the support this team provided. They were professional, communicative, and expeditious in their work, and we were impressed and satisfied with the final outcome.”

Marcela was inspired by her collaborations with Masters of Public Health students  calling communications and public health “a wonderful intertwinement.”



"""
splitter=CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""
)
result =splitter.split_text(text)
print(result[0])
