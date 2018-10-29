import urllib
import requests
import xml.etree.ElementTree as ET


def getAddress(value):
    url = 'http://zip.cgis.biz/xml/zip.php?zn='
    url = url + value
    req = urllib.request.Request(url)

    with urllib.request.urlopen(req) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    es = root.findall('.//value')

    dict = {}
    for e in es:
        dict.update(e.attrib)

    return(dict['state'] + dict['city'])