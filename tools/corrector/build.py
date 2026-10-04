head=open('src/head.part').read()          # title + font links
css=open('src/style.part').read(); body=open('src/body.part').read(); js=open('src/script.part').read()
js=js.replace('__SW__',open('src/sw_words.txt').read()).replace('__EN__',open('src/en_words.txt').read())
open('index.html','w').write(head+"<style>"+css+"</style>\n\n"+body+'\n<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>\n<script>\n'+js+"</script>\n")
print(len(open('index.html').read()))
