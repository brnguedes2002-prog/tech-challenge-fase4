import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from app_config import load_env

class ConfigTests(unittest.TestCase):
    def path(self, content):
        p = Mock()
        p.exists.return_value = True
        p.read_text.return_value = content
        return p

    def test_environment_precedence(self):
        env = {'VOC_MODE': 'local'}
        load_env(self.path('VOC_MODE=databricks\nVOC_MIN_SCORE="0.60"'), env)
        self.assertEqual(env, {'VOC_MODE': 'local', 'VOC_MIN_SCORE': '0.60'})

    def test_no_execution_or_interpolation(self):
        env = {}
        load_env(self.path('DATABRICKS_CONFIG_PROFILE=$(not-a-command)'), env)
        self.assertEqual(env['DATABRICKS_CONFIG_PROFILE'], '$(not-a-command)')

    def test_reject_unrecognized_keys_and_secrets(self):
        for content in ('PATH=x', 'DATABRICKS_TOKEN=x', 'VOC_MODE="broken'):
            with self.assertRaises(ValueError): load_env(self.path(content), {})

    def test_comments_and_missing_file(self):
        env = {}
        load_env(self.path('# comment\n\nVOC_MODE=local'), env)
        self.assertEqual(env['VOC_MODE'], 'local')
        p = self.path('VOC_MODE=databricks'); p.exists.return_value = False
        load_env(p, env)
        self.assertEqual(env['VOC_MODE'], 'local')
