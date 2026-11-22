import { Game } from './Game';
import { createRenderer } from './engine';
import { loadWorld } from './world/WorldData';
import { loadPeople } from './entities/People';

const canvas = document.getElementById('game') as HTMLCanvasElement;
const ui = document.getElementById('ui') as HTMLElement;
ui.innerHTML = '<div class="loading">大阪・天王寺を読み込み中…</div>';

Promise.all([
  loadWorld(),
  loadPeople(),
  createRenderer({ canvas, antialias: true, powerPreference: 'high-performance' }),
])
  .then(([data, , renderer]) => {
    ui.innerHTML = '';
    const game = new Game(canvas, ui, data, renderer);
    game.start();
    // exposed for debugging and automated browser checks
    (window as any).__game = game;
  })
  .catch((err) => {
    ui.innerHTML = '<div class="loading">マップデータを読み込めませんでした: ' + String(err) + '</div>';
  });
