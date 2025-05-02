import './App.css';
import Header from './header.js'; 
import Search from './search.js';
import Chat from './chat.js';

function App() {
  return (
    <div className="App">
      <div className='header'>
        <Header/>
      </div>
      <div className='page'>
        <div className='chat'>
          <Chat/>
        </div>
        <div className='search'>
          <Search/>
        </div>
      </div>
    </div>
  );
}

export default App;
