import { useEffect, useState } from 'react'
import { userApi } from '@/api'
import type { User } from '@acme/core'

export function useUser(id: string) {
  const [user, setUser] = useState<User | null>(null)
  useEffect(() => {
    userApi.fetchUser(id).then(setUser)
  }, [id])
  return user
}
