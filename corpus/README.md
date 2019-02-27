# Obtenir le corpus

Source des données : https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/

```bash
curl -OL https://www.dropbox.com/s/le4yxfijxt0uiia/frwiki-20181001-corpus.xml.bz2
bunzip2 frwiki-20181001-corpus.xml.bz2
perl xml2txt.pl -nomath -notables -nodisambig frwiki-20181001-corpus.xml frwiki-20181001-corpus.txt
wc -l frwiki-20181001-corpus.txt
```
