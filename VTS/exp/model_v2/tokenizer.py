import os
from typing import List, Dict, Union

class CharTokenizer:
    def __init__(self, vocab_file: str):
        """
        初始化字符级别的 Tokenizer
        
        Args:
            vocab_file: 词汇表文件路径，格式为 "字符 ID"，每行一个条目
        """
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        
        # 确保文件存在
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocabulary file {vocab_file} not found")
            
        # 读取词汇表文件
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                char, id_str = line.split()
                char_id = int(id_str)
                self.char2id[char] = char_id
                self.id2char[char_id] = char
                
        # 验证必要的特殊token
        if '<blank>' not in self.char2id:
            raise ValueError("Vocabulary must contain '<blank>' token")
        if '<unk>' not in self.char2id:
            raise ValueError("Vocabulary must contain '<unk>' token")
            
        self.blank_id = self.char2id['<blank>']
        self.unk_id = self.char2id['<unk>']
        self.vocab_size = len(self.char2id)
        
    def encode(self, text: str) -> List[int]:
        """
        将文本转换为token ID列表
        
        Args:
            text: 输入的中文文本
            
        Returns:
            对应的token ID列表
        """
        return [self.char2id.get(char, self.unk_id) for char in text]
        
    def decode(self, ids: List[int], remove_blank: bool = True) -> str:
        """
        将token ID列表转换回文本
        
        Args:
            ids: token ID列表
            remove_blank: 是否移除blank token
            
        Returns:
            解码后的文本
        """
        if remove_blank:
            # 移除连续的重复字符和blank token
            result = []
            prev_id = None
            for id in ids:
                if id == self.blank_id:
                    continue
                if id != prev_id:
                    result.append(self.id2char.get(id, '<unk>'))
                    prev_id = id
            return ''.join(result)
        else:
            # 保留所有token，包括blank
            return ''.join([self.id2char.get(id, '<unk>') for id in ids])
            
    def __len__(self) -> int:
        """返回词汇表大小"""
        return self.vocab_size 