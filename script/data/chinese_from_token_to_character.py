#coding:utf-8
"""
this script can convert 'chinese token file' into 'chinese character file'.

we will check every token, if we find the token is composed by chinese character,
then we will split every chinese character as an independent signal;
else the token will include number or alpha or punctuation ... we still regard it as an independent signal, not change.

"""
__author__ = 'WangQiang'

#usage: input output
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def find_chinese_character(text=''):
	ch_list = text.encode('utf-8')
	index = 0
	find_chi_list = []
	while index < len(ch_list):
		ch = ord(ch_list[index])
		if ch & 0xF0 == 0xE0:
			chi_w = bytes.decode(ch_list[index:index+3])
			#print 'find:',chi_w
			find_chi_list.append(chi_w)
			index += 3
		else:
			find_chi_list = [text]
			return find_chi_list
	return find_chi_list


if len(sys.argv) < 3:
	print 'usage: input output'
	sys.exit(-1)

input = sys.argv[1]
output = sys.argv[2]
#print input,output

out_file = open(output,'w')

with open(input, 'r') as file:
	line_num = 0
	for line in file:
		line_num += 1
		if line_num % 100000 == 0:
			print 'done {} lines ...\r'.format(line_num)
		line = line.strip()
		word_list = line.split()
		# check every token
		for w in word_list:
			chi_w_list = find_chinese_character(w)
			for chi_w in chi_w_list:
				print >>out_file, chi_w,
		print >>out_file
out_file.close()
print

