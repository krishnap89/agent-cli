import { createRoot } from 'react-dom/client'
import { UserList } from '@/components/UserList'
import { repository } from './api'

repository.clear()
createRoot(document.getElementById('root')!).render(<UserList ids={['1', '2']} />)
