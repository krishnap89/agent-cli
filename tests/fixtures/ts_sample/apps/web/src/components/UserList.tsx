import UserCard from './UserCard'
import { useUser } from '../hooks/useUser'
import * as core from '@acme/core'

export const UserList = ({ ids }: { ids: string[] }) => {
  const first = useUser(ids[0])
  const re = /<UserCard\/>"/g
  const client = core.http.createClient()
  const onSelect = (id: string) => console.log(id, re.test(id), client)
  return (
    <>
      {ids.map(id => <UserCard key={id} user={{ id, name: id }} onSelect={onSelect} />)}
      {first && <p>{core.displayName(first)}</p>}
    </>
  )
}
